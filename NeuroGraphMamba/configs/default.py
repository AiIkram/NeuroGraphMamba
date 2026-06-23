"""
configs/default.py
===================
All paths and hyperparameters for NeuroGraphMamba live here, separated from
the model/pipeline code so you never have to touch core.py to change a
dataset path or a hyperparameter.

This is a verbatim extraction of the `Config` class from the original
single-file script — copy it in unchanged if you already have it working,
or adjust the values below to match your environment.
"""
import os
import numpy as np
import torch


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = 1  # force single GPU

    fs = 256
    seg_sec = 2
    target_len = 512
    n_channels = 18      # CHB-MIT default; Siena path patches this at runtime
    n_subbands = 5

    # data
    max_patients        = 24   # all CHB-MIT patients
    max_files_per_pat   = 12
    seg_stride_sz        = 0.5
    seg_stride_nsz        = 1.0
    max_nsz_per_pat     = 1000
    balance_ratio       = 1.0

    # graph
    granger_max_lag     = 3
    prior_weight        = 0.40
    coh_weight          = 0.25
    knn_k               = 4

    # model
    node_dim            = 64
    gcn_dim             = 128
    attn_dim            = 128
    attn_heads          = 4

    # LoRA
    lora_rank           = 4
    lora_alpha          = 8.0

    # training
    batch_size          = 32
    lopo_epochs         = 50
    lopo_patience       = 10
    lopo_warmup         = 8
    lopo_lr             = 3e-4
    lopo_wd             = 1e-4
    dropout             = 0.30
    grad_clip           = 1.0

    # aux losses
    infonce_lam         = 1e-4
    kg_consistency_lam  = 1e-4
    l1_lam              = 1e-5

    # threshold
    threshold_sweep      = np.arange(0.20, 0.81, 0.05).tolist()
    threshold_spec_floor = 0.25

    # TTA
    tta_bn_passes       = 5
    tta_lora_steps      = 15
    tta_lora_lr         = 5e-5
    tta_entropy_div_w   = 0.5

    bootstrap_n         = 500
    bootstrap_ci        = 0.95
    run_lopo            = True
    run_calibration     = True
    run_xai             = True

    # ── PATHS — EDIT THESE FOR YOUR ENVIRONMENT ───────────────────────
    # Never commit real paths containing identifying usernames/orgs if
    # this repo is shared during double-blind review.
    chbmit_path = os.environ.get("CHBMIT_PATH", "/path/to/chbmit/1.0.0")
    siena_path  = os.environ.get("SIENA_PATH",  "/path/to/siena")
    out_path    = os.environ.get("NGM_OUT_PATH", "./outputs")

    # Zero-shot: path to a CHB-MIT-trained checkpoint (set via CLI flag in
    # scripts/run_siena_zero_shot.py, or here directly)
    chbmit_checkpoint_for_zeroshot = None


cfg = Config()
os.makedirs(cfg.out_path, exist_ok=True)
