#!/usr/bin/env python
"""
Run Siena LOPO-CV (train and test on Siena only) — useful as a baseline
or ablation against the CHB-MIT-to-Siena zero-shot transfer result.

Usage:
    python scripts/run_siena_lopo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import cfg
from neurographmamba.core import run_siena_lopo


def main():
    print(f"Loading Siena from {cfg.siena_path}")
    run_siena_lopo(
        siena_path=cfg.siena_path,
        cfg=cfg,
        device=cfg.device,
        out_path=os.path.join(cfg.out_path, "siena_lopo"),
    )


if __name__ == "__main__":
    main()
