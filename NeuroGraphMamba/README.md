# NeuroGraphMamba

Spatial-temporal graph state-space architecture for patient-independent
epileptic seizure detection, evaluated under strict leave-one-patient-out
(LOPO) cross-validation on CHB-MIT and Siena Scalp EEG, with fully
unsupervised test-time adaptation (TTBNorm + per-patient LoRA entropy
minimization + Otsu self-calibrating thresholding).

Companion code for the paper:

> *NeuroGraphMamba: A Spatial-Temporal Graph State-Space Architecture for
> Patient-Independent Epileptic Seizure Detection* (submitted, MICCAI AIMI)

> **Status:** anonymized for double-blind review. Do not add author names,
> affiliations, or links back to a non-anonymous account/profile until
> after the review period.

## Results summary

| Dataset (LOPO-CV)        | AUC-ROC | Sensitivity | Specificity |
|---------------------------|:------:|:-----------:|:-----------:|
| CHB-MIT (24 patients)      | 0.914  | 85.2%       | 88.2%       |
| Siena (13 patients, zero-shot transfer from CHB-MIT) | 0.877 | 83.5% | 81.2% |

Full per-patient numbers, calibration curves, and the learned causal graph
are written to `outputs/` by the scripts below.

## Repository layout

```
NeuroGraphMamba/
├── configs/
│   └── default.py          # all paths + hyperparameters (EDIT THIS for your machine)
├── neurographmamba/
│   ├── __init__.py
│   └── core.py              # model, data loaders, graph construction, TTA, LOPO loops
├── scripts/
│   ├── run_chbmit_lopo.py
│   ├── run_siena_lopo.py
│   └── run_siena_zero_shot.py
├── docs/
│   └── REPRODUCIBILITY.md   # dataset access, known annotation quirks, expected runtime
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

`core.py` intentionally keeps the full pipeline in one module for this
release — it is internally cross-referencing (model classes call graph
builders, training calls TTA, etc.) and splitting it further without being
able to re-run it end-to-end against real data risks introducing subtle
import bugs. The three runnable entry points are the actual public
interface; see `docs/REPRODUCIBILITY.md` if you want to modularize further
post-acceptance.

## Setup

```bash
git clone <repo-url>
cd NeuroGraphMamba
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Edit `configs/default.py`:
- `chbmit_path` → your local CHB-MIT directory (PhysioNet, ODC-BY license)
- `siena_path`  → your local Siena directory (PhysioNet, credentialed access)
- `out_path`    → where results/checkpoints/plots are written

**Do not commit dataset files or trained checkpoints to this repo** —
both datasets require accepting PhysioNet's data use terms individually.

## Usage

```bash
# CHB-MIT LOPO-CV (all 24 patients)
python scripts/run_chbmit_lopo.py

# Siena LOPO-CV (train+test on Siena only — ablation/baseline)
python scripts/run_siena_lopo.py

# Cross-dataset zero-shot transfer: CHB-MIT-trained checkpoint -> Siena
python scripts/run_siena_zero_shot.py --checkpoint outputs/best_LOPO_chb01.pth
```

## Citation

```bibtex
@inproceedings{anonymous2026neurographmamba,
  title     = {NeuroGraphMamba: A Spatial-Temporal Graph State-Space
               Architecture for Patient-Independent Epileptic Seizure
               Detection},
  author    = {Anonymous},
  booktitle = {MICCAI AMAI Workshop},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). This license covers the code in this repository
only — it does **not** grant any rights to the CHB-MIT or Siena Scalp EEG
datasets, which remain governed by their respective PhysioNet data use
agreements.
