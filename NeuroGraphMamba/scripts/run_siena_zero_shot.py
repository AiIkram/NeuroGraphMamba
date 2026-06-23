#!/usr/bin/env python
"""
Cross-dataset transfer: evaluate a CHB-MIT-trained NeuroGraphMamba
checkpoint on Siena, with and without unsupervised test-time adaptation.
Produces Table 2 of the paper.

Usage:
    python scripts/run_siena_zero_shot.py --checkpoint outputs/best_LOPO_chb01.pth
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import cfg
from neurographmamba.core import run_siena_zero_shot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=cfg.chbmit_checkpoint_for_zeroshot,
        help="Path to a CHB-MIT LOPO checkpoint (any best_LOPO_chbXX.pth).",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        raise SystemExit(
            "Provide --checkpoint pointing to a CHB-MIT-trained checkpoint "
            "(produced by scripts/run_chbmit_lopo.py)."
        )

    run_siena_zero_shot(
        siena_path=cfg.siena_path,
        cfg=cfg,
        device=cfg.device,
        chbmit_checkpoint_path=args.checkpoint,
        out_path=os.path.join(cfg.out_path, "siena_zeroshot"),
    )


if __name__ == "__main__":
    main()
