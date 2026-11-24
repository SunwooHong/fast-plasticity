"""Evaluate frozen, keyed, or bias fast plasticity on a TD-MPC2 checkpoint."""
import argparse
import csv
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tdmpc2"))

import torch
from mwm_plastic import BiasCorrection, HebbianLatentResidual

CKPT_URL = "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/{name}?download=true"


def checkpoint_path(task, seed):
    path = ROOT / "checkpoints" / f"{task}-{seed}.pt"
    if path.is_file() and path.stat().st_size > 10_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = CKPT_URL.format(name=path.name)
    tmp = path.with_suffix(".partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def load_agent(task, ckpt):
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from common.parser import parse_cfg
    from common.seed import set_seed
    from envs import make_env
    from tdmpc2 import TDMPC2

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "tdmpc2"), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"task={task}",
                "obs=state",
                "model_size=5",
                "compile=false",
                "enable_wandb=false",
                "save_video=false",
                "hydra/launcher=basic",
                "seed=0",
                f"checkpoint={ckpt}",
            ],
        )
    cfg = parse_cfg(cfg)
    set_seed(0)
    make_env(cfg)
    agent = TDMPC2(cfg)
    agent.load(str(ckpt))
    agent.model.eval()
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    return agent, cfg


def make_plastic(cfg, method):
    if method == "frozen":
        plastic = HebbianLatentResidual(cfg.latent_dim, cfg.action_dim, cfg.simnorm_dim)
        plastic.write = False
    elif method == "keyed":
        plastic = HebbianLatentResidual(cfg.latent_dim, cfg.action_dim, cfg.simnorm_dim)
    elif method == "bias":
        plastic = BiasCorrection(cfg.latent_dim, cfg.simnorm_dim)
    elif method == "constant":
        import json
        from baselines import ConstantKey
        key = torch.tensor(json.loads((ROOT / "configs" / "constant_key.json").read_text()))
        plastic = ConstantKey(cfg.latent_dim, cfg.action_dim, cfg.simnorm_dim, key=key)
    else:
        raise SystemExit(f"unknown method {method}")
    return plastic


def find_attr(root, name):
    seen = set()
    stack = [root]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if hasattr(cur, name):
            return cur
        for attr in ("env", "_env", "wrapped_env"):
            nxt = getattr(cur, attr, None)
            if nxt is not None and nxt is not cur:
                stack.append(nxt)
    raise RuntimeError(name)


class Gain:
    """Commanded action is what the planner outputs. Applied action is command times gain."""

    def __init__(self, env, gain):
        self.env = env
        self.gain = gain

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.float32)
        gain = torch.as_tensor(self.gain, dtype=action.dtype, device=action.device)
        return self.env.step(action * gain)


def scale_torso(env, multiplier):
    if multiplier == 1.0:
        return
    model = find_attr(env, "physics").physics.model
    names = [model.body(i).name for i in range(model.nbody)]
    index = names.index("torso")
    model.body_mass[index] *= multiplier


def run_episode(agent, cfg, plastic, seed, gain, mass):
    from common.seed import set_seed
    from envs import make_env

    set_seed(seed)
    # Environment RNG is cfg.seed. Checkpoint loading stays on seed 0.
    cfg.seed = int(seed)
    plastic.P.zero_()
    plastic.proj_hits = 0
    plastic.steps = 0
    agent.model.mwm = plastic
    agent._prev_mean.zero_()
    env = Gain(make_env(cfg), gain)
    obs = env.reset()
    # Submitted mass runs reset at the nominal mass, then scale the torso.
    scale_torso(env, mass)
    total = 0.0
    shadows = []
    done = False
    for t in range(int(cfg.episode_length)):
        if done:
            break
        action = agent.act(obs, t0=t == 0, eval_mode=True)
        with torch.no_grad():
            z = agent.model.encode(torch.as_tensor(obs, device=agent.device).unsqueeze(0), None)
            a = action.to(agent.device).view(1, -1)
            # Dynamics call sees the commanded action. Gain is applied only in env.step.
            base = agent.model._dynamics(torch.cat([z, a], dim=-1))
            pred = plastic.correct(z, a, base)
        nxt, reward, done, _ = env.step(action)
        with torch.no_grad():
            z_obs = agent.model.encode(torch.as_tensor(nxt, device=agent.device).unsqueeze(0), None)
            saved = plastic.P.detach().clone()
            plastic.P = torch.zeros_like(saved)
            zero = plastic.correct(z, a, base)
            plastic.P = saved
            current = float((pred - z_obs).pow(2).mean())
            shadows.append(float((zero - z_obs).pow(2).mean()) - current)
        # Write uses the prediction that was scored before the environment step.
        plastic.update(z, a, pred, z_obs)
        total += float(reward)
        obs = nxt
        done = bool(done)
    shadow = float(sum(shadows) / len(shadows)) if shadows else 0.0
    norm = float(torch.linalg.vector_norm(plastic.P))
    return total, int(plastic.steps), shadow, norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=("frozen", "keyed", "bias", "constant"), required=True)
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--joint-gains", default="", help="Comma-separated per-joint gains. Overrides --gain.")
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--checkpoint-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "eval.csv")
    args = parser.parse_args()
    if "-" in args.seeds:
        start, end = args.seeds.split("-")
        seeds = range(int(start), int(end) + 1)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    ckpt = checkpoint_path(args.task, args.checkpoint_seed)
    agent, cfg = load_agent(args.task, ckpt)
    plastic = make_plastic(cfg, args.method).to(agent.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new = not args.output.exists()
    with args.output.open("a") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["task", "method", "gain", "mass", "checkpoint_seed", "seed", "return", "steps", "shadow_improvement", "final_norm"])
        gain = args.gain
        if args.joint_gains:
            gain = [float(x) for x in args.joint_gains.split(",")]
        for seed in seeds:
            ret, steps, shadow, norm = run_episode(agent, cfg, plastic, seed, gain, args.mass)
            writer.writerow([args.task, args.method, gain, args.mass, args.checkpoint_seed, seed, ret, steps, shadow, norm])
            handle.flush()
            print(args.task, args.method, seed, ret, flush=True)


if __name__ == "__main__":
    main()
