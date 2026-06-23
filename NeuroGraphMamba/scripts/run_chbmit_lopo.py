#!/usr/bin/env python
"""
Run CHB-MIT LOPO-CV (all 24 patients) end-to-end:
load data -> DWT features -> spatial + causal graphs -> LOPO training
with test-time adaptation -> calibration/plots -> results.csv

Usage:
    python scripts/run_chbmit_lopo.py
"""
import os
import sys
import gc
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from configs.default import cfg
from neurographmamba.core import (
    seed_everything, load_chbmit, extract_node_features,
    build_spatial_adjacency, build_kg_adjacency, NeuroGraphMamba,
    run_lopo, plot_causal_structure, plot_calibration, plot_results,
    CHB_CHANNELS, bootstrap_ci, fmt_ci,
)


def main():
    seed_everything()
    cache = os.path.join(cfg.out_path, "cache")
    os.makedirs(cache, exist_ok=True)

    print(f"Loading CHB-MIT from {cfg.chbmit_path}")
    X_raw, y_all, pid_all = load_chbmit(cfg.chbmit_path, cfg, cache)
    print(f"Patients: {np.unique(pid_all).tolist()}")
    print(f"Class balance: sz={y_all.sum()}  nsz={(y_all == 0).sum()}")

    node_cache = os.path.join(cache, "nodes_all.npy")
    X_nodes = extract_node_features(X_raw, cfg, node_cache)
    del X_raw
    gc.collect()

    n_nodes = X_nodes.shape[1]
    ch_names = CHB_CHANNELS[:n_nodes]

    X_mean_coh = X_nodes[:, :, 0, :].mean(0)
    A_spatial = build_spatial_adjacency(
        ch_names, X_mean_coh, k=cfg.knn_k, coh_w=cfg.coh_weight, fs=cfg.fs
    )

    n_g = min(1000, len(X_nodes))
    idx_g = np.random.choice(len(X_nodes), n_g, replace=False)
    A_kg_glob = build_kg_adjacency(
        X_nodes[idx_g], cfg.granger_max_lag, cfg.prior_weight, cfg.coh_weight
    )

    if cfg.run_xai:
        plot_causal_structure(A_kg_glob, A_spatial, ch_names, cfg.out_path)

    results = run_lopo(X_nodes, y_all, pid_all, A_spatial, ch_names, cfg, cfg.device)

    if not results:
        print("No LOPO results produced — check data loading above.")
        return

    if cfg.run_calibration:
        ece = plot_calibration(results, cfg.out_path)
        with open(os.path.join(cfg.out_path, "ece.json"), "w") as f:
            json.dump({"ece": round(ece, 4)}, f)

    plot_results(results, cfg.out_path)
    pd.DataFrame(results).to_csv(os.path.join(cfg.out_path, "results.csv"), index=False)

    print("\n" + "=" * 70 + "\nCHB-MIT FINAL RESULTS\n" + "=" * 70)
    for name, vals, pct in [
        ("Balanced Acc",  [r["bacc"] * 100 for r in results], True),
        ("Sensitivity",   [r["sens"] * 100 for r in results], True),
        ("Specificity",   [r["spec"] * 100 for r in results], True),
        ("AUC-ROC",       [r["auc"]        for r in results], False),
    ]:
        m, lo, hi = bootstrap_ci(vals, cfg.bootstrap_n, cfg.bootstrap_ci)
        print(f"  {name:<18}: {fmt_ci(m, lo, hi, pct)}")

    print(f"\nAll outputs -> {cfg.out_path}")


if __name__ == "__main__":
    main()
