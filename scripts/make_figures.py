"""Write the paired-return figure from results/episode_returns.csv."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_results import figure, load

ROOT = Path(__file__).resolve().parents[1]
figure(load(), ROOT / "results" / "paired_bias_keyed.svg")
print("wrote", ROOT / "results" / "paired_bias_keyed.svg")
