"""Shared repo-relative locations for the analysis scripts."""
import sys
from pathlib import Path

REPO     = Path(__file__).resolve().parents[1]
RESULTS  = REPO / 'results'
FIGS     = REPO / 'figures'
TABLES   = REPO / 'tables'
PSA_DIR  = REPO / 'psa'
PSA_DATA = PSA_DIR / 'data'
CKPT_DIR = REPO / 'checkpoints'


def add_psa_to_path():
    """Make the psa modules importable from the analysis scripts."""
    p = str(PSA_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
