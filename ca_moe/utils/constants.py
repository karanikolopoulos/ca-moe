from pathlib import Path

PROJECT = "Cost-Aware Mixture of Experts"

ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = ROOT / "data" / "conf"

DATA_DIR = ROOT / "data" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
