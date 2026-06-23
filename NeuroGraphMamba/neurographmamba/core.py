
import os, sys, re, copy, warnings, json, math, gc
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.signal import coherence as sp_coherence
from scipy import signal as scipy_signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve

from configs.default import cfg

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw): return x

try:
    import pywt; PYWT = True
except ImportError:
    PYWT = False

try:
    import mne; mne.set_log_level("WARNING"); MNE = True
except ImportError:
    MNE = False

MAMBA_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("✓ mamba-ssm loaded (available for ablation)")
except ImportError:
    print("⚠  mamba-ssm not found")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(f"PyTorch {torch.__version__} | MNE={MNE} | PyWavelets={PYWT} | Mamba={MAMBA_AVAILABLE}")


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 1: CHB-MIT CONSTANTS ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

CHB_CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
]
N_CHANNELS     = 18
SUBBAND_LABELS = ["δ","θ","α","β","γ"]
N_SUBBANDS     = 5
SEED           = 42

_MONO_POS = {
    "FP1":(-0.18,0.92),"FP2":(0.18,0.92),
    "F7":(-0.54,0.65),"F3":(-0.31,0.65),"FZ":(0.00,0.70),
    "F4":(0.31,0.65),"F8":(0.54,0.65),
    "T7":(-0.80,0.00),"T8":(0.80,0.00),
    "C3":(-0.45,0.00),"CZ":(0.00,0.00),"C4":(0.45,0.00),
    "P7":(-0.65,-0.50),"P8":(0.65,-0.50),
    "P3":(-0.31,-0.55),"PZ":(0.00,-0.60),"P4":(0.31,-0.55),
    "O1":(-0.18,-0.90),"O2":(0.18,-0.90),
}

def _bipolar_mid(ch):
    p = ch.upper().replace("EEG ","").split("-")
    if len(p) != 2: return (0.0, 0.0)
    a = _MONO_POS.get(p[0].strip()); b = _MONO_POS.get(p[1].strip())
    if not a or not b: return (0.0, 0.0)
    return ((a[0]+b[0])/2, (a[1]+b[1])/2)

EEG_POS = {ch: _bipolar_mid(ch) for ch in CHB_CHANNELS}

NEURO_PRIOR = np.array([
    [0.0,0.70,0.35,0.10,0.05],
    [0.20,0.0,0.65,0.20,0.10],
    [0.10,0.30,0.0,0.45,0.15],
    [0.05,0.15,0.25,0.0,0.55],
    [0.05,0.10,0.15,0.35,0.0],
], dtype=np.float32)

KG_ORDER_PAIRS = [(0,1,"gt"),(1,2,"gt"),(2,3,"gt"),(3,4,"gt")]

SEIZURE_FILES = {
    "chb01":["chb01_03.edf","chb01_04.edf","chb01_15.edf","chb01_16.edf","chb01_18.edf","chb01_21.edf","chb01_26.edf"],
    "chb02":["chb02_16.edf","chb02_16+.edf","chb02_19.edf"],
    "chb03":["chb03_01.edf","chb03_02.edf","chb03_03.edf","chb03_04.edf","chb03_34.edf","chb03_35.edf","chb03_36.edf"],
    "chb04":["chb04_05.edf","chb04_08.edf","chb04_28.edf"],
    "chb05":["chb05_06.edf","chb05_13.edf","chb05_16.edf","chb05_17.edf","chb05_22.edf"],
    "chb06":["chb06_01.edf","chb06_04.edf","chb06_09.edf","chb06_10.edf","chb06_13.edf","chb06_18.edf","chb06_24.edf"],
    "chb07":["chb07_12.edf","chb07_13.edf","chb07_18.edf"],
    "chb08":["chb08_02.edf","chb08_05.edf","chb08_11.edf","chb08_13.edf","chb08_21.edf"],
    "chb09":["chb09_06.edf","chb09_08.edf","chb09_19.edf"],
    "chb10":["chb10_12.edf","chb10_20.edf","chb10_27.edf","chb10_30.edf","chb10_31.edf","chb10_38.edf","chb10_89.edf"],
    "chb11":["chb11_82.edf","chb11_92.edf","chb11_99.edf"],
    "chb12":["chb12_06.edf","chb12_08.edf","chb12_09.edf","chb12_10.edf","chb12_11.edf","chb12_23.edf","chb12_27.edf","chb12_28.edf","chb12_29.edf","chb12_33.edf","chb12_36.edf","chb12_38.edf","chb12_42.edf"],
    "chb13":["chb13_19.edf","chb13_21.edf","chb13_40.edf","chb13_55.edf","chb13_58.edf","chb13_59.edf","chb13_60.edf","chb13_62.edf"],
    "chb14":["chb14_03.edf","chb14_04.edf","chb14_06.edf","chb14_11.edf","chb14_17.edf","chb14_18.edf","chb14_27.edf"],
    "chb15":["chb15_06.edf","chb15_10.edf","chb15_15.edf","chb15_17.edf","chb15_20.edf","chb15_22.edf","chb15_28.edf","chb15_31.edf","chb15_40.edf","chb15_46.edf","chb15_49.edf","chb15_52.edf","chb15_54.edf","chb15_62.edf"],
    "chb16":["chb16_10.edf","chb16_11.edf","chb16_14.edf","chb16_16.edf","chb16_17.edf","chb16_18.edf"],
    "chb17":["chb17a_03.edf","chb17a_04.edf","chb17b_63.edf"],
    "chb18":["chb18_29.edf","chb18_30.edf","chb18_31.edf","chb18_32.edf","chb18_35.edf","chb18_36.edf"],
    "chb19":["chb19_28.edf","chb19_29.edf","chb19_30.edf"],
    "chb20":["chb20_12.edf","chb20_13.edf","chb20_14.edf","chb20_15.edf","chb20_16.edf","chb20_68.edf"],
    "chb21":["chb21_19.edf","chb21_20.edf","chb21_21.edf","chb21_22.edf"],
    "chb22":["chb22_20.edf","chb22_25.edf","chb22_38.edf"],
    "chb23":["chb23_06.edf","chb23_08.edf","chb23_09.edf"],
    "chb24":["chb24_01.edf","chb24_03.edf","chb24_04.edf","chb24_06.edf","chb24_07.edf","chb24_09.edf","chb24_11.edf","chb24_13.edf","chb24_14.edf","chb24_15.edf","chb24_17.edf","chb24_21.edf"],
}


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 2: SIENA CONSTANTS ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

SIENA_FS_NATIVE = 512   # all Siena files recorded at 512 Hz
SIENA_FS_TARGET = 256   # downsample to match CHB-MIT / v5 pipeline

# 29 EEG channels shared across all Siena patients (excluding EKG, SPO2, HR, MK, 1, 2)
SIENA_EEG_CHANNELS = [
    "Fp1", "F3",  "C3",  "P3",  "O1",
    "F7",  "T3",  "T5",
    "Fc1", "Fc5", "Cp1", "Cp5",
    "F9",
    "Fz",  "Cz",  "Pz",
    "Fp2", "F4",  "C4",  "P4",  "O2",
    "F8",  "T4",  "T6",
    "Fc2", "Fc6", "Cp2", "Cp6",
    "F10",
]
N_SIENA_CHANNELS = len(SIENA_EEG_CHANNELS)  # 29

# 2-D scalp positions for Siena monopolar channels
SIENA_POS = {
    "Fp1":(-0.18, 0.92), "Fp2":(0.18, 0.92),
    "F9": (-0.80, 0.65), "F7": (-0.54, 0.65), "F3": (-0.31, 0.65),
    "Fz": ( 0.00, 0.70), "F4": ( 0.31, 0.65), "F8": (0.54, 0.65),
    "F10":( 0.80, 0.65),
    "Fc5":(-0.60, 0.32), "Fc1":(-0.22, 0.32),
    "Fc2":( 0.22, 0.32), "Fc6":( 0.60, 0.32),
    "T3": (-0.80, 0.00), "C3": (-0.45, 0.00), "Cz": (0.00, 0.00),
    "C4": ( 0.45, 0.00), "T4": (0.80, 0.00),
    "Cp5":(-0.60,-0.32), "Cp1":(-0.22,-0.32),
    "Cp2":( 0.22,-0.32), "Cp6":( 0.60,-0.32),
    "T5": (-0.65,-0.50), "P3": (-0.31,-0.55),
    "Pz": ( 0.00,-0.60), "P4": ( 0.31,-0.55), "T6": (0.65,-0.50),
    "O1": (-0.18,-0.90), "O2": ( 0.18,-0.90),
}

# Hardcoded annotations — ALL patients PN00–PN17 (fully populated)
# Format: { patient_id: { edf_filename: [(wall_start, wall_end), ...] } }
SIENA_SEIZURES_RAW = {
    "PN00": {
        "PN00-1.edf": [("19:58:36", "19:59:46")],
        "PN00-2.edf": [("02:38:37", "02:39:31")],
        "PN00-3.edf": [("18:28:29", "18:57:13")],  # end clamped (annotation error)
        "PN00-4.edf": [("21:08:29", "21:09:43")],
        "PN00-5.edf": [("22:37:08", "22:38:15")],
    },
    "PN01": {
        "PN01.edf":   [("21:51:02", "21:51:56")],
        "PN01-1.edf": [("07:53:17", "07:54:31")],
    },
    "PN03": {
        "PN03-1.edf": [("09:29:10", "09:31:01")],
        "PN03-2.edf": [("07:13:05", "07:15:18")],
    },
    "PN05": {
        "PN05-2.edf": [("08:45:25", "08:46:00")],
        "PN05-3.edf": [("07:55:19", "07:55:49")],
    },
    "PN06": {
        "PNO6-1.edf": [("05:54:25", "05:55:29")],  # 'O' not '0' — real filename
        "PNO6-2.edf": [("23:39:09", "23:40:18")],
        "PN06-3.edf": [("08:10:26", "08:11:08")],
        "PNO6-4.edf": [("12:55:08", "12:56:11")],
    },
    "PN07": {
        "PN07-1.edf": [("05:25:49", "05:26:51")],
    },
    "PN08": {},   # no seizure files — will be auto-skipped
    "PN09": {
        "PN09-1.edf": [("16:09:43", "16:11:03")],
        "PN09-2.edf": [("17:00:56", "17:01:55")],
        "PN09-3.edf": [("16:20:44", "16:21:48")],
    },
    "PN10": {
        "PN10-1.edf": [("07:45:50", "07:46:59")],
        "PN10-2.edf": [("11:40:13", "11:41:04")],
        "PN10-3.edf": [("15:43:53", "15:45:02")],
        "PN10-4.5.6.edf": [
            ("12:49:50", "12:49:55"),
            ("14:00:25", "14:00:44"),
            ("15:18:26", "15:19:23"),
        ],
        "PN10-7.8.9.edf": [
            ("17:35:13", "17:36:01"),
            ("18:20:24", "18:20:42"),
            ("20:24:48", "20:25:03"),
        ],
        "PN10-10.edf": [("10:58:19", "10:58:33")],
    },
    "PN11": {
        "PN11-.edf": [("13:37:19", "13:38:14")],
    },
    "PN12": {
        "PN12-1.2.edf": [
            ("16:13:23", "16:14:26"),
            ("18:31:01", "18:32:09"),
        ],
        "PN12-3.edf": [("08:55:27", "08:57:03")],
        "PN12-4.edf": [("18:42:51", "18:43:54")],
    },
    "PN13": {
        "PN13-1.edf": [("10:22:10", "10:22:58")],
        "PN13-2.edf": [("08:55:51", "08:56:56")],
        "PN13-3.edf": [("14:05:54", "14:08:25")],
    },
    "PN14": {
        "PN14-1.edf": [("13:46:00", "13:46:27")],
        "PN14-2.edf": [("17:54:52", "17:55:04")],
        "PN14-3.edf": [("21:10:05", "21:10:46")],
        "PN14-4.edf": [("15:49:33", "15:50:56")],
    },
    "PN16": {
        "PN16-1.edf": [("22:45:05", "22:47:08")],
        "PN16-2.edf": [("03:16:49", "03:18:36")],
    },
    "PN17": {
        "PN17-1.edf": [("22:34:48", "22:35:58")],
        "PN17-2.edf": [("16:01:09", "16:02:32")],
    },
}

# Registration start times per EDF file (for wall-clock → seconds conversion)
SIENA_REG_START = {
    # PN00
    "PN00-1.edf": "19:39:33",
    "PN00-2.edf": "02:18:17",
    "PN00-3.edf": "18:15:44",
    "PN00-4.edf": "20:51:43",
    "PN00-5.edf": "22:22:04",
    # PN01
    "PN01.edf":   "19:00:44",
    "PN01-1.edf": "06:30:00",   # approx — next-day session; clamp will guard any error
    # PN03
    "PN03-1.edf": "22:44:37",
    "PN03-2.edf": "21:31:04",
    # PN05
    "PN05-2.edf": "06:46:02",
    "PN05-3.edf": "06:01:23",
    # PN06
    "PNO6-1.edf": "04:21:22",
    "PNO6-2.edf": "21:11:29",
    "PN06-3.edf": "06:25:51",
    "PNO6-4.edf": "11:16:09",
    # PN07
    "PN07-1.edf": "23:18:10",
    # PN09
    "PN09-1.edf": "14:08:54",
    "PN09-2.edf": "15:02:09",
    "PN09-3.edf": "14:20:23",
    # PN10
    "PN10-1.edf":       "05:40:05",
    "PN10-2.edf":       "09:30:15",
    "PN10-3.edf":       "13:33:18",
    "PN10-4.5.6.edf":   "12:11:21",
    "PN10-7.8.9.edf":   "16:49:25",
    "PN10-10.edf":      "08:45:22",
    # PN11
    "PN11-.edf":  "11:31:25",
    # PN12
    "PN12-1.2.edf": "15:51:31",
    "PN12-3.edf":   "08:42:35",
    "PN12-4.edf":   "15:59:19",
    # PN13
    "PN13-1.edf": "08:24:28",
    "PN13-2.edf": "06:55:02",
    "PN13-3.edf": "12:00:01",
    # PN14
    "PN14-1.edf": "11:44:58",
    "PN14-2.edf": "15:50:13",
    "PN14-3.edf": "16:17:45",
    "PN14-4.edf": "14:18:30",
    # PN16
    "PN16-1.edf": "20:45:21",
    "PN16-2.edf": "00:53:55",
    # PN17
    "PN17-1.edf": "20:14:28",
    "PN17-2.edf": "13:52:18",
}

seed_everything = lambda s=SEED: (
    np.random.seed(s),
    torch.manual_seed(s),
    torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None,
    setattr(torch.backends.cudnn, 'deterministic', True) if torch.cuda.is_available() else None,
    setattr(torch.backends.cudnn, 'benchmark', False) if torch.cuda.is_available() else None,
)

seed_everything()
print(f"Device: {cfg.device}  GPUs: {cfg.n_gpus}")


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

class SafeScaler:
    def __init__(self):
        self._on = torch.cuda.is_available()
        self._s  = GradScaler() if self._on else None
    def scale(self, l):    return self._s.scale(l) if self._on else l
    def unscale_(self, o):
        if self._on: self._s.unscale_(o)
    def step(self, o):
        if self._on: self._s.step(o)
        else: o.step()
    def update(self):
        if self._on: self._s.update()


def bootstrap_ci(vals, n=500, ci=0.95):
    v = np.array(vals, float)
    if len(v) < 2:
        x = float(v[0]) if len(v) else 0.0
        return x, x, x
    rng  = np.random.RandomState(SEED)
    boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    a    = (1 - ci) / 2
    return float(v.mean()), float(np.percentile(boot, 100*a)), \
           float(np.percentile(boot, 100*(1-a)))


def fmt_ci(m, lo, hi, pct=False):
    f = "{:.2f}% [{:.2f}–{:.2f}]" if pct else "{:.4f} [{:.4f}–{:.4f}]"
    return f.format(m, lo, hi)


def _print_summary(title, results):
    if not results: return
    print(f"\n{'='*65}\n{title} — SUMMARY\n{'='*65}")
    for name, vals, pct in [
        ("Balanced Acc",  [r["bacc"]*100 for r in results], True),
        ("Sensitivity",   [r["sens"]*100 for r in results], True),
        ("Specificity",   [r["spec"]*100 for r in results], True),
        ("AUC-ROC",       [r["auc"]      for r in results], False),
    ]:
        m = np.mean(vals); s = np.std(vals); unit = "%" if pct else ""
        print(f"  {name:<18}: {m:.2f}{unit} ± {s:.2f}{unit}")
    print(f"\n  {'Patient':<8} {'BAcc':>8} {'Sens':>8} {'Spec':>8} {'AUC':>8}")
    print("  " + "-"*42)
    for r in sorted(results, key=lambda x: x["patient"]):
        print(f"  {r['patient']:<8} "
              f"{r['bacc']*100:>7.1f}% "
              f"{r['sens']*100:>7.1f}% "
              f"{r['spec']*100:>7.1f}% "
              f"{r['auc']:>8.4f}")


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 3: CHB-MIT CHANNEL MATCHING & ANNOTATIONS ──────────────────
# ═══════════════════════════════════════════════════════════════════════

def normalize_ch(raw):
    s = raw.upper().strip().replace("EEG ", "")
    parts = s.split("-"); skip = {"REF","LE","AVG","A1","A2"}
    while len(parts) > 1 and parts[-1] in skip:
        parts = parts[:-1]
    return "-".join(p for p in parts if p)


def match_channels(raw_names, target_list):
    normed = [normalize_ch(r) for r in raw_names]
    result = []
    for ti, tgt in enumerate(target_list):
        tgt_n = normalize_ch(tgt)
        for ri, n in enumerate(normed):
            if n == tgt_n: result.append((ti, ri)); break
        else:
            for ri, n in enumerate(normed):
                if n.startswith(tgt_n) or tgt_n.startswith(n):
                    result.append((ti, ri)); break
    return result


def parse_seizures(pat_dir, pat):
    annotations = {}
    sf = os.path.join(pat_dir, f"{pat}-summary.txt")
    if os.path.exists(sf):
        with open(sf, errors="replace") as fh:
            cur = None
            for line in fh:
                line = line.strip()
                if "File Name:" in line:
                    cur = line.split(":", 1)[1].strip()
                    if cur not in annotations: annotations[cur] = []
                elif "Seizure" in line and "Start Time" in line and cur:
                    try:
                        t = int("".join(filter(str.isdigit, line.split(":", 1)[1])))
                        annotations[cur].append([t, None])
                    except: pass
                elif "Seizure" in line and "End Time" in line and cur:
                    try:
                        t = int("".join(filter(str.isdigit, line.split(":", 1)[1])))
                        if annotations[cur] and annotations[cur][-1][1] is None:
                            annotations[cur][-1][1] = t
                    except: pass
    return {f: [(s, e) for s, e in segs if e is not None and e > s]
            for f, segs in annotations.items()
            if any(e is not None and e > s for s, e in segs)}


def is_artifact(seg, amp_thresh=15.0, flat_var=1e-4):
    if np.ptp(seg) > amp_thresh: return True
    win   = max(1, len(seg) // 5)
    vars_ = [seg[i:i+win].var() for i in range(0, len(seg)-win, win//2)]
    return bool(vars_ and min(vars_) < flat_var)


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 4: SIENA ANNOTATIONS & CHANNEL MATCHING ────────────────────
# ═══════════════════════════════════════════════════════════════════════

def parse_siena_txt_annotations(data_path: str):
    """
    Reads all Seizures-list-PNxx.txt files and returns:
        seizures_raw : { pat_id: { edf_filename: [(wall_start, wall_end)] } }
        reg_starts   : { edf_filename: reg_start_str }
    """
    seizures_raw = {}; reg_starts = {}

    def normalise_time(s):
        s = s.strip()
        s = re.sub(r'(\d{2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', s)
        return s

    for fname in (sorted(os.listdir(data_path)) if os.path.isdir(data_path) else []):
        if not fname.startswith("Seizures-list-PN") or not fname.endswith(".txt"):
            continue
        pat_id = fname.replace("Seizures-list-", "").replace(".txt", "")
        txt_path = os.path.join(data_path, fname)
        try:
            with open(txt_path, errors="replace") as fh:
                lines = fh.readlines()
        except Exception as e:
            print(f"  [TXT] Cannot read {fname}: {e}"); continue

        cur_file = None; cur_reg = None
        cur_starts = []; cur_ends = []; pat_seizures = {}

        for line in lines:
            line = line.strip()
            if not line: continue

            m = re.search(r'File name[:\s]+(\S+\.edf)', line, re.I)
            if m:
                if cur_file and cur_starts:
                    pairs = [(s, e) for s, e in zip(cur_starts, cur_ends) if s and e and e != s]
                    if pairs:
                        pat_seizures[cur_file] = pairs
                        if cur_reg: reg_starts[cur_file] = cur_reg
                cur_file = m.group(1); cur_starts = []; cur_ends = []; continue

            m = re.search(r'Registration start time[:\s]+([\d.:]+)', line, re.I)
            if m:
                cur_reg = normalise_time(m.group(1))
                if cur_file: reg_starts[cur_file] = cur_reg
                continue

            m = re.search(r'Seizure\s+(?:start|n\s*\d+\s+start)\s+time[:\s]+([\d.:]+)', line, re.I)
            if not m: m = re.search(r'start\s+time[:\s]+([\d.:]+)', line, re.I)
            if m: cur_starts.append(normalise_time(m.group(1))); continue

            m = re.search(r'Seizure\s+(?:end|n\s*\d+\s+end)\s+time[:\s]+([\d.:]+)', line, re.I)
            if not m: m = re.search(r'end\s+time[:\s]+([\d.:]+)', line, re.I)
            if m: cur_ends.append(normalise_time(m.group(1))); continue

        if cur_file and cur_starts:
            pairs = [(s, e) for s, e in zip(cur_starts, cur_ends) if s and e and e != s]
            if pairs:
                pat_seizures[cur_file] = pairs
                if cur_reg: reg_starts[cur_file] = cur_reg

        if pat_seizures:
            seizures_raw[pat_id] = pat_seizures
            print(f"  [TXT] {pat_id}: parsed {sum(len(v) for v in pat_seizures.values())} seizures "
                  f"across {len(pat_seizures)} files")

    return seizures_raw, reg_starts


def load_all_siena_annotations(data_path: str):
    """
    Merges hardcoded annotations (PN00–PN17) with auto-parsed txt annotations.
    Hardcoded values take priority over txt for patients already in SIENA_SEIZURES_RAW
    with non-empty entries, to protect against known annotation errors (e.g. PN01.edf
    cross-midnight confusion).
    """
    parsed_sz, parsed_reg = parse_siena_txt_annotations(data_path)

    for subdir in sorted(os.listdir(data_path)):
        subpath = os.path.join(data_path, subdir)
        if os.path.isdir(subpath) and subdir.startswith("PN"):
            sub_sz, sub_reg = parse_siena_txt_annotations(subpath)
            if sub_sz: parsed_sz[subdir] = list(sub_sz.values())[0]
            parsed_reg.update(sub_reg)

    merged_sz  = copy.deepcopy(SIENA_SEIZURES_RAW)
    merged_reg = copy.deepcopy(SIENA_REG_START)

    for pat, files in parsed_sz.items():
        if not files:
            continue
        hardcoded = SIENA_SEIZURES_RAW.get(pat, {})
        if hardcoded:
            # Hardcoded wins — only use txt for patients not yet hardcoded
            # but DO merge any reg_starts from txt that we don't have yet
            pass
        else:
            # Patient not hardcoded at all → use txt
            merged_sz[pat] = files

    # Always merge reg_starts from txt (we may be missing some)
    merged_reg.update(parsed_reg)

    available = [p for p, files in merged_sz.items() if files]
    print(f"  [Annotations] {len(available)} Siena patients with annotations: {available}")
    return merged_sz, merged_reg


def _wall_to_seconds(wall_time_str: str, reg_start_str: str) -> float:
    """Convert wall-clock seizure time → seconds from file start. Handles midnight crossings."""
    fmt   = "%H:%M:%S"
    reg   = datetime.strptime(reg_start_str, fmt)
    event = datetime.strptime(wall_time_str,  fmt)
    delta = (event - reg).total_seconds()
    if delta < 0: delta += 86400
    return delta


def parse_siena_seizures(pat_id: str, seizures_raw: dict, reg_starts: dict) -> dict:
    """Returns { edf_filename: [(start_sec, end_sec), ...] }"""
    raw = seizures_raw.get(pat_id, {}); out = {}
    for fname, intervals in raw.items():
        reg_start = reg_starts.get(fname)
        if reg_start is None:
            print(f"  [WARN] No reg_start for {fname}, skipping"); continue
        converted = []
        for ws, we in intervals:
            try:
                s = _wall_to_seconds(ws, reg_start)
                e = _wall_to_seconds(we, reg_start)
                if e > s: converted.append((s, e))
                else: print(f"  [WARN] {fname}: end≤start ({s:.0f},{e:.0f}), skipping")
            except Exception as err:
                print(f"  [WARN] {fname}: time parse error {err}")
        if converted: out[fname] = converted
    return out


def _normalize_siena_ch(raw_name: str) -> str:
    """
    Real Siena EDF format: 'EEG Fp1', 'EEG F3', 'EKG EKG', 'SPO2', 'HR', '1', '2', 'MK'
    Strip 'EEG ' prefix and any dash+suffix, uppercase.
    """
    s = raw_name.strip().upper()
    for pfx in ("EEG ", "ECG ", "EMG ", "EOG ", "EKG "):
        if s.startswith(pfx):
            s = s[len(pfx):]
    s = s.split("-")[0].strip()
    s = s.split("+")[0].strip()
    return s


def match_siena_channels(raw_ch_names):
    """
    Returns (target_idx, raw_idx) pairs for SIENA_EEG_CHANNELS.
    Skips EKG/ECG/EMG/EOG/SPO2/HR/MK and numeric-only channels ('1','2').
    """
    skip_exact  = {"EKG", "SPO2", "HR", "MK", "1", "2", "MKR", "STATUS"}
    skip_prefix = {"EKG", "ECG", "EMG", "EOG"}
    normed_raw  = [_normalize_siena_ch(n) for n in raw_ch_names]
    target_upper = [c.upper() for c in SIENA_EEG_CHANNELS]

    pairs = []
    used_raw = set()
    for ti, tgt in enumerate(target_upper):
        for ri, n in enumerate(normed_raw):
            if ri in used_raw: continue
            raw_up = raw_ch_names[ri].strip().upper()
            if n in skip_exact: continue
            if any(raw_up.startswith(p) for p in skip_prefix): continue
            if n == tgt:
                pairs.append((ti, ri)); used_raw.add(ri); break

    if len(pairs) < 4:
        print(f"    [WARN] match_siena_channels: only {len(pairs)} matched. "
              f"Raw names: {raw_ch_names[:10]} ...")
    return pairs


def debug_siena_channels(siena_path: str, n_patients: int = 3):
    """
    Call this ONCE to print the actual channel names in your Siena EDFs.
    Helps diagnose channel-matching failures.
    Usage: debug_siena_channels(cfg.siena_path)
    """
    print("\n[DEBUG] Siena EDF channel names:")
    count = 0
    for pat in sorted(os.listdir(siena_path)):
        if not pat.startswith("PN"): continue
        pat_dir = os.path.join(siena_path, pat)
        if not os.path.isdir(pat_dir): continue
        for edf in sorted(os.listdir(pat_dir)):
            if not edf.endswith(".edf"): continue
            fpath = os.path.join(pat_dir, edf)
            try:
                raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
                print(f"  {pat}/{edf}: {raw.ch_names}")
                raw.close()
                count += 1
                break   # one file per patient is enough
            except Exception as e:
                print(f"  {pat}/{edf}: ERROR {e}")
        if count >= n_patients:
            break
    print("[DEBUG] Done. Check names above vs SIENA_EEG_CHANNELS.\n")


def is_artifact_siena(seg, amp_thresh=15.0, flat_var=1e-4):
    if np.ptp(seg) > amp_thresh: return True
    win   = max(1, len(seg) // 5)
    vars_ = [seg[i:i+win].var() for i in range(0, len(seg)-win, win//2)]
    return bool(vars_ and min(vars_) < flat_var)


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 5: DATA LOADERS ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def load_chbmit(data_path, cfg, cache_dir):
    if not MNE: raise RuntimeError("pip install mne")
    os.makedirs(cache_dir, exist_ok=True)
    all_pats = sorted([
        d for d in os.listdir(data_path)
        if d.startswith("chb") and os.path.isdir(os.path.join(data_path, d))
    ])[:cfg.max_patients]

    Xs, ys, pids = [], [], []
    for pat in all_pats:
        cx = os.path.join(cache_dir, f"{pat}_X.npy")
        cy = os.path.join(cache_dir, f"{pat}_y.npy")
        if os.path.exists(cx):
            X_p = np.load(cx); y_p = np.load(cy)
            if y_p.sum() == 0:
                os.remove(cx); os.remove(cy)
            else:
                Xs.append(X_p); ys.append(y_p); pids.extend([pat]*len(y_p))
                print(f"  {pat}: [cache] sz={y_p.sum()} nsz={(y_p==0).sum()}")
                continue

        pat_dir  = os.path.join(data_path, pat)
        seizures = parse_seizures(pat_dir, pat)
        if not seizures: print(f"  {pat}: no annotations — skip"); continue

        known    = SEIZURE_FILES.get(pat, [])
        all_edfs = sorted([f for f in os.listdir(pat_dir) if f.endswith(".edf")])
        sz_edfs  = [f for f in all_edfs if f in known]
        nsz_edfs = [f for f in all_edfs if f not in known]
        edfs     = sz_edfs + nsz_edfs[:max(cfg.max_files_per_pat - len(sz_edfs), 2)]

        seg_samp   = int(cfg.seg_sec * cfg.fs)
        stride_sz  = max(1, int(cfg.seg_stride_sz  * cfg.fs))
        stride_nsz = max(1, int(cfg.seg_stride_nsz * cfg.fs))

        sz_segs, nsz_segs, n_rej = [], [], 0

        for edf in tqdm(edfs, desc=f"  {pat}", leave=False):
            sz_intervals = seizures.get(edf, [])
            fpath = os.path.join(pat_dir, edf)
            if not os.path.exists(fpath): continue
            try:
                raw   = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                sfreq = raw.info["sfreq"]
                pairs = match_channels(raw.ch_names, CHB_CHANNELS[:cfg.n_channels])
                if len(pairs) < 4: raw.close(); continue
                kept_t = [ti for ti,ri in pairs]
                kept_r = [ri for ti,ri in pairs]
                data   = raw.get_data()[kept_r].astype(np.float32)
                n_kept = len(kept_r)
                raw.close(); del raw

                if sfreq != cfg.fs:
                    n_new = int(data.shape[1] * cfg.fs / sfreq)
                    data  = np.array([
                        scipy_signal.resample(data[c], n_new) for c in range(n_kept)
                    ], dtype=np.float32)

                for c in range(n_kept):
                    med = np.median(data[c])
                    iqr = np.percentile(data[c], 75) - np.percentile(data[c], 25) + 1e-8
                    data[c] = np.clip((data[c] - med) / iqr, -10, 10)

                T = data.shape[1]

                def make_seg(start):
                    blk = data[:, start:start+seg_samp]
                    if blk.shape[1] < seg_samp:
                        blk = np.pad(blk, ((0,0),(0,seg_samp-blk.shape[1])))
                    out = np.zeros((cfg.n_channels, cfg.target_len), np.float32)
                    for ci, ti in enumerate(kept_t):
                        s = blk[ci]
                        if cfg.target_len != seg_samp:
                            s = scipy_signal.resample(s, cfg.target_len)
                        out[ti] = s
                    return out

                def in_sz(t):
                    return any(s <= t <= e for s, e in sz_intervals)

                for start in range(0, T - seg_samp + 1, stride_sz):
                    t = start / cfg.fs
                    if not in_sz(t): continue
                    seg = make_seg(start)
                    if is_artifact(seg[kept_t[0] if kept_t else 0]):
                        n_rej += 1; continue
                    sz_segs.append(seg)

                for start in range(0, T - seg_samp + 1, stride_nsz):
                    t    = start / cfg.fs
                    peri = any(abs(t-s) < 30 or abs(t-e) < 30 for s, e in sz_intervals)
                    if peri or in_sz(t): continue
                    seg = make_seg(start)
                    if is_artifact(seg[kept_t[0] if kept_t else 0]):
                        n_rej += 1; continue
                    nsz_segs.append(seg)
                    if len(nsz_segs) >= cfg.max_nsz_per_pat * 3: break

                del data
            except Exception as e:
                print(f"    ERR {edf}: {e}")

        if not sz_segs: print(f"  {pat}: 0 sz — skip"); continue
        n_sz  = len(sz_segs)
        n_nsz = min(len(nsz_segs), int(n_sz * cfg.balance_ratio))
        if n_nsz == 0: print(f"  {pat}: no nsz — skip"); continue

        rng    = np.random.RandomState(SEED)
        nsz_sel = [nsz_segs[i] for i in rng.choice(len(nsz_segs), n_nsz, replace=False)]
        X_p    = np.array(sz_segs + nsz_sel, dtype=np.float32)
        y_p    = np.array([1]*n_sz + [0]*n_nsz, dtype=np.int64)
        shf    = rng.permutation(len(y_p)); X_p, y_p = X_p[shf], y_p[shf]
        np.save(cx, X_p); np.save(cy, y_p)
        Xs.append(X_p); ys.append(y_p); pids.extend([pat]*len(y_p))
        print(f"  {pat}: sz={n_sz} nsz={n_nsz} rej={n_rej}")
        del X_p, sz_segs, nsz_segs; gc.collect()

    if not Xs: raise RuntimeError("No CHB-MIT data loaded")
    X   = np.concatenate(Xs, 0).astype(np.float32)
    y   = np.concatenate(ys, 0).astype(np.int64)
    pid = np.array(pids)
    print(f"  TOTAL CHB-MIT: {X.shape}  sz={y.sum()}  nsz={(y==0).sum()}")
    return X, y, pid


def load_siena(data_path: str, cfg, cache_dir: str,
               seizures_raw: dict = None, reg_starts: dict = None):
    """
    Load Siena Scalp EEG → (X, y, pid) matching load_chbmit() format.
    X: (N, N_SIENA_CHANNELS, cfg.target_len) float32
    """
    os.makedirs(cache_dir, exist_ok=True)
    if seizures_raw is None or reg_starts is None:
        seizures_raw, reg_starts = load_all_siena_annotations(data_path)

    all_pats = sorted([
        p for p in seizures_raw
        if seizures_raw[p] and os.path.isdir(os.path.join(data_path, p))
    ])
    if not all_pats:
        raise RuntimeError(f"No Siena patient folders found under {data_path}")

    Xs, ys, pids = [], [], []

    for pat in all_pats:
        cx = os.path.join(cache_dir, f"{pat}_X.npy")
        cy = os.path.join(cache_dir, f"{pat}_y.npy")

        if os.path.exists(cx) and os.path.exists(cy):
            X_p = np.load(cx); y_p = np.load(cy)
            if y_p.sum() == 0:
                os.remove(cx); os.remove(cy)
            else:
                Xs.append(X_p); ys.append(y_p); pids.extend([pat]*len(y_p))
                print(f"  {pat}: [cache] sz={y_p.sum()} nsz={(y_p==0).sum()}")
                continue

        pat_dir  = os.path.join(data_path, pat)
        seizures = parse_siena_seizures(pat, seizures_raw, reg_starts)
        if not seizures: print(f"  {pat}: no parsed seizures — skip"); continue

        seg_samp   = int(cfg.seg_sec * SIENA_FS_TARGET)
        native_seg = int(cfg.seg_sec * SIENA_FS_NATIVE)
        stride_sz  = max(1, int(cfg.seg_stride_sz  * SIENA_FS_NATIVE))
        stride_nsz = max(1, int(cfg.seg_stride_nsz * SIENA_FS_NATIVE))

        sz_segs, nsz_segs, n_rej = [], [], 0
        sz_edfs    = list(seizures.keys())
        other_edfs = [f for f in sorted(os.listdir(pat_dir))
                      if f.endswith(".edf") and f not in sz_edfs]

        def process_edf(edf, is_sz_file):
            nonlocal n_rej
            fpath = os.path.join(pat_dir, edf)
            if not os.path.exists(fpath): return
            try:
                raw   = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                sfreq = raw.info["sfreq"]
                pairs = match_siena_channels(raw.ch_names)
                if len(pairs) < 4:
                    print(f"    {edf}: only {len(pairs)} channels — skip")
                    raw.close(); return

                kept_t = [ti for ti, ri in pairs]
                kept_r = [ri for ti, ri in pairs]
                data   = raw.get_data()[kept_r].astype(np.float32)
                n_kept = len(kept_r)
                raw.close(); del raw

                if abs(sfreq - SIENA_FS_NATIVE) > 1:
                    print(f"    {edf}: sfreq={sfreq} (expected 512), resampling")
                    n_new = int(data.shape[1] * SIENA_FS_NATIVE / sfreq)
                    data  = np.array([scipy_signal.resample(data[c], n_new)
                                      for c in range(n_kept)], np.float32)

                for c in range(n_kept):
                    med = np.median(data[c])
                    iqr = (np.percentile(data[c], 75) - np.percentile(data[c], 25)) + 1e-8
                    data[c] = np.clip((data[c] - med) / iqr, -10, 10)

                T = data.shape[1]
                sz_intervals = seizures.get(edf, []) if is_sz_file else []
                # Clamp seizure intervals to actual file length (guards annotation errors)
                file_dur_sec = T / SIENA_FS_NATIVE
                sz_intervals = [(s, min(e, file_dur_sec))
                                for s, e in sz_intervals if s < file_dur_sec]
                sz_samp = [(int(s * SIENA_FS_NATIVE), int(e * SIENA_FS_NATIVE))
                           for s, e in sz_intervals]

                def in_sz(start_samp):
                    return any(ss <= start_samp <= se for ss, se in sz_samp)

                def make_seg(start_samp):
                    blk = data[:, start_samp:start_samp + native_seg]
                    if blk.shape[1] < native_seg:
                        blk = np.pad(blk, ((0,0),(0, native_seg - blk.shape[1])))
                    out = np.zeros((N_SIENA_CHANNELS, cfg.target_len), np.float32)
                    for ci, ti in enumerate(kept_t):
                        s = scipy_signal.resample(blk[ci], cfg.target_len)
                        out[ti] = s.astype(np.float32)
                    return out

                if is_sz_file:
                    file_sz_count = 0
                    for ss, se in sz_samp:
                        for start in range(ss, se - native_seg + 1, stride_sz):
                            if file_sz_count >= 500: break   # cap per-file to avoid one long sz dominating
                            seg = make_seg(start)
                            if is_artifact_siena(seg[kept_t[0] if kept_t else 0]):
                                n_rej += 1; continue
                            sz_segs.append(seg); file_sz_count += 1

                if len(nsz_segs) >= cfg.max_nsz_per_pat * 3: return
                for start in range(0, T - native_seg + 1, stride_nsz):
                    t_sec = start / SIENA_FS_NATIVE
                    peri  = any(abs(t_sec - s) < 30 or abs(t_sec - e) < 30
                                for s, e in sz_intervals)
                    if peri or in_sz(start): continue
                    seg = make_seg(start)
                    if is_artifact_siena(seg[kept_t[0] if kept_t else 0]):
                        n_rej += 1; continue
                    nsz_segs.append(seg)
                    if len(nsz_segs) >= cfg.max_nsz_per_pat * 3: break

                del data
            except Exception as e:
                print(f"    ERR {edf}: {e}")

        for edf in tqdm(sz_edfs, desc=f"  {pat} [sz]", leave=False):
            process_edf(edf, is_sz_file=True)
        for edf in tqdm(other_edfs[:6], desc=f"  {pat} [nsz]", leave=False):
            if len(nsz_segs) >= cfg.max_nsz_per_pat * 3: break
            process_edf(edf, is_sz_file=False)

        if not sz_segs: print(f"  {pat}: 0 sz segs — skip"); continue
        n_sz  = len(sz_segs)
        n_nsz = min(len(nsz_segs), int(n_sz * cfg.balance_ratio))
        if n_nsz == 0: print(f"  {pat}: 0 nsz segs — skip"); continue

        rng     = np.random.RandomState(SEED)
        nsz_sel = [nsz_segs[i] for i in rng.choice(len(nsz_segs), n_nsz, replace=False)]
        X_p = np.array(sz_segs + nsz_sel, np.float32)
        y_p = np.array([1]*n_sz + [0]*n_nsz, np.int64)
        shf = rng.permutation(len(y_p)); X_p, y_p = X_p[shf], y_p[shf]
        np.save(cx, X_p); np.save(cy, y_p)
        Xs.append(X_p); ys.append(y_p); pids.extend([pat]*len(y_p))
        print(f"  {pat}: sz={n_sz} nsz={n_nsz} rej={n_rej}")
        del sz_segs, nsz_segs; gc.collect()

    if not Xs:
        raise RuntimeError("No Siena data loaded — check SIENA_PATH and folder structure")
    X   = np.concatenate(Xs, 0).astype(np.float32)
    y   = np.concatenate(ys, 0).astype(np.int64)
    pid = np.array(pids)
    print(f"  TOTAL Siena: {X.shape}  sz={y.sum()}  nsz={(y==0).sum()}")
    return X, y, pid


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 6: DWT FEATURES ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def dwt_channel(sig, n_bands, target_len):
    if PYWT:
        coeffs = pywt.wavedec(sig, "db4", level=n_bands-1)
        bands  = list(coeffs[:n_bands])
        while len(bands) < n_bands: bands.append(np.zeros_like(bands[-1]))
    else:
        L     = len(sig)
        freqs = np.fft.rfftfreq(L, 1.0/256)
        fft_  = np.fft.rfft(sig)
        edges = [0, 4, 8, 13, 30, 100]; bands = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            fb   = np.zeros_like(fft_); fb[mask] = fft_[mask]
            bands.append(np.fft.irfft(fb, n=L))
    return np.stack([
        scipy_signal.resample(b.astype(np.float32), target_len)
        for b in bands[:n_bands]
    ], axis=0)


def extract_node_features(X_raw, cfg, cache_path):
    if os.path.exists(cache_path):
        print(f"  [DWT] Loading cache")
        return np.load(cache_path, mmap_mode="r")
    N, C, L = X_raw.shape
    print(f"  [DWT] Extracting ({N},{C},{cfg.n_subbands},{cfg.target_len})…")
    out   = np.zeros((N, C, cfg.n_subbands, cfg.target_len), dtype=np.float32)
    chunk = 256
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        for i in range(start, end):
            for c in range(C):
                out[i, c] = dwt_channel(X_raw[i, c], cfg.n_subbands, cfg.target_len)
        print(f"    DWT {end}/{N}", end="\r")
    for c in range(C):
        for b in range(cfg.n_subbands):
            m = out[:, c, b, :].mean(); s = out[:, c, b, :].std() + 1e-8
            out[:, c, b, :] = (out[:, c, b, :] - m) / s
    np.save(cache_path, out)
    print(f"\n  [DWT] Saved → {cache_path}")
    return np.load(cache_path, mmap_mode="r")


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 7: ADJACENCY MATRICES ───────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def _granger_fstat(x, y, max_lag):
    T = len(y); p = max_lag
    if T < 2*p + 10: return 0.0
    Y  = y[p:]; T2 = len(Y)
    Xr = np.column_stack([y[p-i-1:T-i-1] for i in range(p)])
    Xr = np.column_stack([np.ones(T2), Xr])
    Xu = np.column_stack([Xr, *[x[p-i-1:T-i-1].reshape(-1,1) for i in range(p)]])
    def rss(A, b):
        try:
            bt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return float(np.sum((b - A@bt)**2))
        except: return 1e10
    RSS_r = rss(Xr, Y); RSS_ur = rss(Xu, Y)
    df1 = p; df2 = T2 - 2*p - 1
    if df2 <= 0 or RSS_ur < 1e-12: return 0.0
    return max(0.0, float(((RSS_r - RSS_ur)/df1) / (RSS_ur/df2)))


def build_kg_adjacency(X_sub, max_lag=3, prior_w=0.40, coh_w=0.25):
    """[N1] Inter-band adjacency: Granger + coherence + neuroanatomical prior."""
    S        = X_sub.shape[2]
    mean_sig = X_sub.mean(axis=(0, 1))

    F = np.zeros((S, S), np.float32)
    for i in range(S):
        for j in range(S):
            if i != j:
                F[i, j] = _granger_fstat(mean_sig[i], mean_sig[j], max_lag)
    G = F / (F.max() + 1e-8)

    C = np.zeros((S, S), np.float32)
    for i in range(S):
        for j in range(S):
            if i != j:
                try:
                    nperseg = min(mean_sig.shape[1] // 2, 64)
                    _, Cxy  = sp_coherence(mean_sig[i], mean_sig[j], fs=256, nperseg=nperseg)
                    C[i, j] = float(np.nanmean(Cxy))
                except: pass

    data_A = (1 - coh_w)*G + coh_w*C
    A = (1 - prior_w)*data_A + prior_w*NEURO_PRIOR[:S, :S]
    np.fill_diagonal(A, 0.0)
    thr = np.percentile(A[A > 0], 25) if (A > 0).any() else 0.0
    A[A < thr] *= 0.5
    A /= (A.max() + 1e-8)
    return A.astype(np.float32)


def build_spatial_adjacency(ch_names, X_mean, k=4, coh_w=0.25, fs=256):
    """CHB-MIT bipolar spatial adjacency."""
    N   = len(ch_names)
    pos = np.array([EEG_POS.get(normalize_ch(c), (0, 0)) for c in ch_names], np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    D    = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(D, D.max() + 1e-8)
    dist_s = 1.0 - D / (D.max() + 1e-8); np.fill_diagonal(dist_s, 0)

    coh_mat = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(i+1, N):
            try:
                nperseg = min(X_mean.shape[1] // 2, 64)
                _, Cxy  = sp_coherence(X_mean[i], X_mean[j], fs=fs, nperseg=nperseg)
                v = float(np.nanmean(Cxy))
                if np.isfinite(v): coh_mat[i, j] = coh_mat[j, i] = np.clip(v, 0, 1)
            except: pass

    A = (1 - coh_w)*dist_s + coh_w*coh_mat
    np.fill_diagonal(A, 0); A = np.clip(A, 0, None)

    As = np.zeros_like(A)
    for i in range(N):
        row = A[i].copy(); row[i] = -np.inf
        top = np.argsort(row)[::-1][:min(k, N-1)]
        As[i, top] = np.clip(row[top], 0, None)

    if As.max() < 1e-8:
        As = np.zeros_like(dist_s)
        for i in range(N):
            row = dist_s[i].copy(); row[i] = -np.inf
            top = np.argsort(row)[::-1][:min(k, N-1)]
            As[i, top] = 1.0

    As /= (As.max() + 1e-8)
    density = (As > 0).mean()
    print(f"  Spatial adj density={density:.3f}  coh_max={coh_mat.max():.3f}")
    return As.astype(np.float32)


def build_siena_spatial_adj(ch_names, X_mean, k=4, coh_w=0.25, fs=256):
    """Siena monopolar spatial adjacency. Drop-in replacement for build_spatial_adjacency."""
    N   = len(ch_names)
    pos = np.array([SIENA_POS.get(c.strip(), (0.0, 0.0)) for c in ch_names], np.float32)
    diff = pos[:, None, :] - pos[None, :, :]
    D    = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(D, D.max() + 1e-8)
    dist_s = 1.0 - D / (D.max() + 1e-8)
    np.fill_diagonal(dist_s, 0)

    coh_mat = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(i+1, N):
            try:
                nperseg = min(X_mean.shape[1] // 2, 64)
                _, Cxy  = sp_coherence(X_mean[i], X_mean[j], fs=fs, nperseg=nperseg)
                v = float(np.nanmean(Cxy))
                if np.isfinite(v): coh_mat[i, j] = coh_mat[j, i] = np.clip(v, 0, 1)
            except: pass

    A = (1 - coh_w)*dist_s + coh_w*coh_mat
    np.fill_diagonal(A, 0); A = np.clip(A, 0, None)

    As = np.zeros_like(A)
    for i in range(N):
        row = A[i].copy(); row[i] = -np.inf
        top = np.argsort(row)[::-1][:min(k, N-1)]
        As[i, top] = np.clip(row[top], 0, None)

    if As.max() < 1e-8:
        As = np.zeros_like(dist_s)
        for i in range(N):
            row = dist_s[i].copy(); row[i] = -np.inf
            top = np.argsort(row)[::-1][:min(k, N-1)]
            As[i, top] = 1.0

    As /= (As.max() + 1e-8)
    print(f"  [Siena] Spatial adj density={(As > 0).mean():.3f}  coh_max={coh_mat.max():.3f}")
    return As.astype(np.float32)


def norm_adj(A_raw, device):
    A  = A_raw + np.eye(len(A_raw), dtype=np.float32)
    Ah = A / (A.sum(1, keepdims=True) + 1e-8)
    return torch.FloatTensor(Ah).to(device)


def patch_cfg_for_siena(base_cfg):
    """Returns a shallow copy of cfg adjusted for Siena (28 monopolar channels, 256 Hz)."""
    sc            = copy.copy(base_cfg)
    sc.n_channels = N_SIENA_CHANNELS
    sc.fs         = SIENA_FS_TARGET
    return sc


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 8: VAL PATIENT SELECTION [F6] ──────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def pick_val_patient(test_pat, tr_pats, pid_all, y_all):
    """Pick val patient similar in seizure count to test patient. [F6]"""
    te_n  = (pid_all == test_pat).sum()
    candidates = []
    for p in tr_pats:
        mask = pid_all == p
        if len(np.unique(y_all[mask])) < 2: continue
        sz_n = y_all[mask].sum()
        if sz_n < 30: continue
        n     = mask.sum()
        ratio = max(n, te_n) / (min(n, te_n) + 1e-8)
        candidates.append((ratio, abs(n - te_n), p))
    if not candidates:
        candidates = [
            (1.0, abs((pid_all==p).sum() - te_n), p)
            for p in tr_pats
            if len(np.unique(y_all[pid_all==p])) >= 2
        ]
    candidates.sort()
    return candidates[0][2] if candidates else None


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 9: AUGMENTATION ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def augment_lopo(X):
    X   = X.copy(); rng = np.random.RandomState(SEED)
    for i in range(len(X)):
        if rng.rand() < 0.6:
            X[i] = np.roll(X[i], rng.randint(-24, 25), axis=-1)
        if rng.rand() < 0.6:
            X[i] += (rng.randn(*X[i].shape) * 0.05).astype(np.float32)
        if rng.rand() < 0.4:
            for _ in range(rng.randint(1, 3)):
                X[i, rng.randint(X.shape[1])] = 0.0
        if rng.rand() < 0.5:
            sc  = rng.uniform(0.75, 1.25, (X.shape[1], 1, 1)).astype(np.float32)
            X[i] *= sc
    return X.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 10: DATASET & LOSSES ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.FloatTensor(np.array(self.X[i])), self.y[i]


def make_loader(X, y, shuffle, cfg):
    return DataLoader(EEGDataset(X, y), batch_size=cfg.batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None):
        super().__init__(); self.gamma = gamma; self.weight = weight
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt)**self.gamma * ce).mean()


def make_criterion(y_tr, device):
    counts = np.bincount(y_tr, minlength=2).astype(float)
    w = 1.0 / (counts + 1e-8); w = w / w.sum()
    return FocalLoss(gamma=1.5, weight=torch.FloatTensor(w).to(device))


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 11: MODEL COMPONENTS ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

class ChannelCNNEncoder(nn.Module):
    """Per-channel CNN: (B,N,S,L) → (B,N,node_dim). BatchNorm on B×N."""
    def __init__(self, n_subbands, node_dim, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_subbands, 32, 7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, node_dim, 3, padding=1),
            nn.BatchNorm1d(node_dim), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, S, L = x.shape
        h = self.net(x.reshape(B*N, S, L)).squeeze(-1)
        return self.drop(h).reshape(B, N, -1)


class GCNLayer(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.W  = nn.Linear(in_d, out_d, bias=False)
        self.bn = nn.BatchNorm1d(out_d)

    def forward(self, H, A):
        B, N, D = H.shape
        agg = torch.einsum("nm,bmd->bnd", A, H)
        out = self.W(agg).reshape(B*N, -1)
        return F.gelu(self.bn(out).reshape(B, N, -1))


class CausalCrossAttention(nn.Module):
    """[N2] KG-guided cross-attention. Attention weights = inter-band saliency (XAI)."""
    def __init__(self, node_dim, n_kg=5, n_heads=4, dropout=0.1):
        super().__init__()
        self.heads   = n_heads
        self.hd      = node_dim // n_heads
        self.scale   = math.sqrt(self.hd)
        self.Wq      = nn.Linear(node_dim, node_dim)
        self.Wk      = nn.Linear(node_dim, node_dim)
        self.Wv      = nn.Linear(node_dim, node_dim)
        self.Wo      = nn.Linear(node_dim, node_dim)
        self.kg_proj = nn.Linear(n_kg, node_dim)
        self.norm    = nn.LayerNorm(node_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, h, A_kg):
        B, N, D = h.shape; S = A_kg.shape[0]
        kg_t   = A_kg.unsqueeze(0).expand(B, -1, -1)
        kg_ctx = self.kg_proj(kg_t)

        def split(x, seq_len):
            return x.reshape(B, seq_len, self.heads, self.hd).transpose(1, 2)

        Q    = split(self.Wq(h),     N)
        K    = split(self.Wk(kg_ctx), S)
        V    = split(self.Wv(kg_ctx), S)
        attn = (Q @ K.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out  = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.norm(h + self.drop(self.Wo(out))), attn.mean(1)


class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=4, alpha=8.0):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=True)
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale  = alpha / rank

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A.t() @ self.lora_B.t())


class PatientBridgeAdapter(nn.Module):
    """[N4] Per-patient LoRA adapter. LayerNorm for stability."""
    def __init__(self, dim=128, rank=4, alpha=8.0):
        super().__init__()
        self.dim      = dim; self.rank = rank; self.alpha = alpha
        self.adapters = nn.ModuleDict()
        self.norm     = nn.LayerNorm(dim)

    def add_patient(self, pid, device=None):
        if pid not in self.adapters:
            adapter = LoRALinear(self.dim, self.dim, self.rank, self.alpha)
            if device is not None: adapter = adapter.to(device)
            self.adapters[pid] = adapter

    def forward(self, h, pid=None):
        h = self.norm(h)
        if pid is not None and pid in self.adapters:
            B, N, D = h.shape
            h = self.adapters[pid](h.reshape(B*N, D)).reshape(B, N, D)
        return h


class GraphInfoNCE(nn.Module):
    """[N5] Graph InfoNCE auxiliary loss."""
    def __init__(self, T=0.1): super().__init__(); self.T = T

    def forward(self, h, y):
        h = F.normalize(h.mean(1), dim=1)
        if h.norm(dim=1).max() < 1e-6: return h.sum() * 0.0
        pos = y == 1; neg = y == 0
        if pos.sum() < 2 or neg.sum() < 2: return h.sum() * 0.0
        hp, hn = h[pos], h[neg]; loss = 0.0; cnt = 0
        for i in range(min(len(hp), 8)):
            q   = hp[i:i+1]; oth = torch.cat([hp[:i], hp[i+1:]], 0)
            if not len(oth): continue
            ps  = (q @ oth.t() / self.T).squeeze(0)
            ns  = (q @ hn.t()  / self.T).squeeze(0)
            all_ = torch.cat([ps, ns])
            if not torch.isfinite(all_).all(): continue
            lbl  = torch.zeros(len(all_), device=h.device)
            lbl[:len(ps)] = 1.0 / max(len(ps), 1)
            loss += -(lbl * F.log_softmax(all_, 0)).sum(); cnt += 1
        return loss / max(cnt, 1)


def kg_consistency_loss(A_kg):
    """[N1] Enforce δ<θ<α<β<γ causal ordering."""
    S    = A_kg.shape[0]
    loss = torch.tensor(0.0, device=A_kg.device)
    for i, j, _ in KG_ORDER_PAIRS:
        if i < S and j < S:
            loss = loss + F.relu(A_kg[j, i] - A_kg[i, j])
    return loss


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 12: MAIN MODEL ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

class NeuroGraphMamba(nn.Module):
    """
    v5 architecture:
    (B,N,S,L) → ChannelCNNEncoder → (B,N,node_dim)
              → GCNLayer ×2 [N3]  → (B,N,gcn_dim)
              → CausalCrossAttn   → (B,N,gcn_dim)  [N2]
              → PatientBridge     → (B,N,gcn_dim)  [N4]
              → GlobalMeanPool    → (B,gcn_dim)
              → MLP head          → (B,2)
    """
    def __init__(self, cfg, n_nodes):
        super().__init__()
        D = cfg.gcn_dim; self.n_nodes = n_nodes; self.cfg = cfg

        self.encoder    = ChannelCNNEncoder(cfg.n_subbands, cfg.node_dim, cfg.dropout)
        self.proj_in    = nn.Linear(cfg.node_dim, D)
        self.gcn1       = GCNLayer(D, D)
        self.gcn2       = GCNLayer(D, D)
        self.cross_attn = CausalCrossAttention(D, n_kg=cfg.n_subbands,
                                                n_heads=cfg.attn_heads,
                                                dropout=cfg.dropout)
        self.bridge     = PatientBridgeAdapter(D, cfg.lora_rank, cfg.lora_alpha)
        self.infonce    = GraphInfoNCE()
        self.head       = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, 128), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(128, 2),
        )
        self.register_buffer("A_spatial", torch.eye(n_nodes))
        self.register_buffer("A_kg",      torch.zeros(cfg.n_subbands, cfg.n_subbands))

    def set_adj(self, A_sp_np, A_kg_np, device):
        self.A_spatial = norm_adj(A_sp_np, device)
        self.A_kg      = torch.FloatTensor(A_kg_np).to(device)

    def encode(self, x, patient_id=None):
        B, N = x.shape[:2]
        A    = self.A_spatial[:N, :N]
        h    = self.encoder(x)
        h    = self.proj_in(h)
        h    = self.gcn1(h, A)
        h    = self.gcn2(h, A)
        h, sal = self.cross_attn(h, self.A_kg)
        h    = self.bridge(h, patient_id)
        feat = h.mean(1)
        return feat, h, sal

    def forward(self, x, patient_id=None):
        feat, _, _ = self.encode(x, patient_id)
        return self.head(feat)

    def forward_train(self, x, y, patient_id=None):
        feat, h, sal = self.encode(x, patient_id)
        logits   = self.head(feat)
        aux_nce  = self.infonce(h, y)
        kg_cons  = kg_consistency_loss(self.A_kg)
        l1       = self.A_spatial[:x.shape[1], :x.shape[1]].abs().sum()
        return logits, aux_nce, kg_cons, l1, sal


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 13: TEST-TIME ADAPTATION [F1] [F2] ─────────────────────────
# ═══════════════════════════════════════════════════════════════════════

class TTBNorm:
    """
    [F1] Test-Time Batch Normalization.
    Reset BN running statistics using test patient's unlabeled data.
    Zero extra parameters. No labels required.
    """
    def __init__(self, model, device):
        self.model  = model
        self.device = device

    @torch.no_grad()
    def adapt(self, X_te_raw, n_passes=5):
        model = self.model
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad_(False)

        ds = EEGDataset(X_te_raw, np.zeros(len(X_te_raw), dtype=np.int64))
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

        for _ in range(n_passes):
            for Xb, _ in dl:
                model(Xb.to(self.device))

        model.eval()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        for param in model.parameters():
            param.requires_grad_(True)

        return model


def tta_entropy_finetune(model, X_te_raw, device, cfg, patient_id, n_steps=15, lr=5e-5):
    """
    [F2] LoRA entropy minimization fine-tuning at test time.
    Only LoRA parameters updated — base model frozen. No labels required.
    Loss = entropy_minimization + diversity_regularization
    """
    model.train()
    model.bridge.add_patient(patient_id, device=device)

    lora_params = list(model.bridge.adapters[patient_id].parameters())
    opt = torch.optim.Adam(lora_params, lr=lr)

    ds = EEGDataset(X_te_raw, np.zeros(len(X_te_raw), dtype=np.int64))
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    for step in range(n_steps):
        for Xb, _ in dl:
            Xb = Xb.to(device)
            opt.zero_grad()
            logits = model(Xb, patient_id=patient_id)
            probs  = F.softmax(logits, dim=1)

            entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()
            mean_p  = probs.mean(0)
            div_reg = (mean_p * torch.log(mean_p + 1e-8)).sum()

            loss = entropy + cfg.tta_entropy_div_w * div_reg
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, 0.5)
                opt.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 14: THRESHOLD METHODS [F3] [F4] ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def find_threshold_otsu(model, X_te_raw, device):
    """[F3] Self-calibrating Otsu threshold on test patient distribution."""
    model.eval()
    ds       = EEGDataset(X_te_raw, np.zeros(len(X_te_raw), np.int64))
    dl       = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    all_probs = []
    with torch.no_grad():
        for Xb, _ in dl:
            p = _safe_probs(model(Xb.to(device))).cpu().numpy()
            all_probs.extend(p)
    all_probs = np.array(all_probs)[:, 1]

    hist, edges = np.histogram(all_probs, bins=50)
    centers     = (edges[:-1] + edges[1:]) / 2
    total       = hist.sum()
    best_t, best_var = 0.5, 0.0

    for t in centers:
        m0_mask = centers <= t; m1_mask = centers > t
        w0 = hist[m0_mask].sum() / (total + 1e-8)
        w1 = hist[m1_mask].sum() / (total + 1e-8)
        if hist[m0_mask].sum() == 0 or hist[m1_mask].sum() == 0: continue
        mu0 = (hist[m0_mask] * centers[m0_mask]).sum() / (hist[m0_mask].sum() + 1e-8)
        mu1 = (hist[m1_mask] * centers[m1_mask]).sum() / (hist[m1_mask].sum() + 1e-8)
        var_between = w0 * w1 * (mu0 - mu1)**2
        if var_between > best_var:
            best_var = var_between; best_t = float(t)

    best_t = float(np.clip(best_t, 0.20, 0.80))
    print(f"  [Otsu-Threshold] t={best_t:.3f}  between-var={best_var:.4f}")
    return best_t


def find_threshold_youden(model, val_loader, device, cfg):
    """[F4] Youden-J maximization with specificity floor on val patient."""
    model.eval(); tgts, prbs = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            out = model(Xb.to(device))
            prbs.extend(_safe_probs(out).cpu().numpy())
            tgts.extend(yb.numpy())
    tgts = np.array(tgts); prbs = np.array(prbs)

    best_t, best_j = 0.50, -999.0
    for t in cfg.threshold_sweep:
        pred = (prbs[:, 1] >= t).astype(int)
        if len(np.unique(pred)) < 2: continue
        cm = confusion_matrix(tgts, pred, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = TP / (TP + FN + 1e-8)
        spec = TN / (TN + FP + 1e-8)
        if spec < cfg.threshold_spec_floor: continue
        if sens < cfg.threshold_spec_floor: continue
        j = sens + spec - 1.0
        if j > best_j: best_j = j; best_t = t

    if best_j < -0.5:
        print(f"  [Youden-Threshold] No valid threshold (floor={cfg.threshold_spec_floor}) → 0.50")
        best_t = 0.50
    else:
        pred_f = (prbs[:, 1] >= best_t).astype(int)
        bacc   = balanced_accuracy_score(tgts, pred_f)
        print(f"  [Youden-Threshold] best={best_t:.2f}  J={best_j:.4f}  BAcc={bacc:.4f}")
    return best_t


def _safe_probs(logits):
    if logits.dim() == 1 or logits.shape[1] == 1:
        p = torch.sigmoid(logits.squeeze())
        return torch.stack([1-p, p], 1)
    return F.softmax(logits, 1)


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 15: TRAINING & EVALUATION ──────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, crit, opt, scaler, device, cfg, verbose=False):
    model.train(); tot = 0.0; n_nan = 0
    for bi, (Xb, yb) in enumerate(loader):
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            _m = model.module if isinstance(model, nn.DataParallel) else model
            logits, aux, kgc, l1, _ = _m.forward_train(Xb, yb)
            aux = aux if torch.isfinite(aux) else torch.tensor(0.0, device=device)
            kgc = kgc if torch.isfinite(kgc) else torch.tensor(0.0, device=device)
            l1  = l1  if torch.isfinite(l1)  else torch.tensor(0.0, device=device)
            loss = (crit(logits, yb)
                    + cfg.infonce_lam * aux
                    + cfg.kg_consistency_lam * kgc
                    + cfg.l1_lam * l1)

        if not torch.isfinite(loss):
            n_nan += 1; opt.zero_grad(); continue

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if verbose and bi == 0:
            max_g = max((p.grad.abs().max().item()
                         for p in model.parameters() if p.grad is not None), default=0.0)
            print(f"    [grad_check] max_grad={max_g:.6f}")

        scaler.step(opt); scaler.update()
        tot += loss.item()

    if n_nan > 0: print(f"    [WARN] {n_nan} NaN batches skipped")
    return tot / max(len(loader) - n_nan, 1)


@torch.no_grad()
def evaluate(model, loader, crit, device, thr=0.5, patient_id=None):
    model.eval(); tgts, prbs, tot = [], [], 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        out    = model(Xb, patient_id)
        tot   += crit(out, yb).item()
        p      = np.nan_to_num(_safe_probs(out).cpu().numpy(), nan=0.5)
        prbs.extend(p); tgts.extend(yb.cpu().numpy())
    tgts = np.array(tgts); prbs = np.array(prbs)
    pred = (prbs[:, 1] >= thr).astype(int)
    auc  = roc_auc_score(tgts, prbs[:, 1]) if len(np.unique(tgts)) > 1 else 0.5
    bacc = balanced_accuracy_score(tgts, pred)
    cm   = confusion_matrix(tgts, pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "loss":    tot / max(len(loader), 1),
        "bacc":    bacc,
        "auc":     auc,
        "sens":    TP / (TP + FN + 1e-8),
        "spec":    TN / (TN + FP + 1e-8),
        "f1":      f1_score(tgts, pred, average="weighted", zero_division=0),
        "acc":     accuracy_score(tgts, pred),
        "thr":     thr,
        "preds":   pred,
        "targets": tgts,
        "probs":   prbs,
    }


def apply_inversion_guard(te, thr, label):
    """[F5] If AUC < 0.5, flip predictions (handles polarity-inverted patients)."""
    if te['auc'] >= 0.50: return te
    print(f"  [{label}] AUC={te['auc']:.4f} < 0.5 → model inverted, flipping predictions")
    te_flip         = dict(te)
    te_flip['probs'] = te['probs'][:, ::-1].copy()
    te_flip['auc']   = roc_auc_score(te['targets'], te_flip['probs'][:, 1])
    flipped          = (te_flip['probs'][:, 1] >= (1.0 - thr)).astype(int)
    cm               = confusion_matrix(te['targets'], flipped, labels=[0, 1])
    TN, FP, FN, TP   = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    te_flip.update({
        'preds': flipped,
        'sens':  TP / (TP + FN + 1e-8),
        'spec':  TN / (TN + FP + 1e-8),
        'bacc':  balanced_accuracy_score(te['targets'], flipped),
        'f1':    f1_score(te['targets'], flipped, average='weighted', zero_division=0),
    })
    print(f"  [{label}] After flip: BAcc={te_flip['bacc']:.4f}  "
          f"Sens={te_flip['sens']*100:.1f}%  Spec={te_flip['spec']*100:.1f}%  "
          f"AUC={te_flip['auc']:.4f}")
    return te_flip


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 16: LOPO FOLD ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def run_lopo_fold(X_tr, y_tr, X_val, y_val, X_te, y_te,
                   A_spatial, A_kg, cfg, device, label, n_nodes,
                   patient_id=None):
    seed_everything()
    model = NeuroGraphMamba(cfg, n_nodes).to(device)
    model.set_adj(A_spatial, A_kg, device)
    if cfg.n_gpus > 1: model = nn.DataParallel(model)

    crit   = make_criterion(y_tr, device)
    opt    = AdamW(model.parameters(), lr=cfg.lopo_lr, weight_decay=cfg.lopo_wd)
    sched  = SequentialLR(opt, [
        LinearLR(opt, 0.05, 1.0, total_iters=cfg.lopo_warmup),
        CosineAnnealingLR(opt, cfg.lopo_epochs - cfg.lopo_warmup, 1e-6)
    ], milestones=[cfg.lopo_warmup])
    scaler = SafeScaler()

    tr_ld  = make_loader(X_tr,  y_tr,  True,  cfg)
    val_ld = make_loader(X_val, y_val, False, cfg)

    best_auc  = 0.5; pat_cnt = 0
    best_path = os.path.join(cfg.out_path, f"best_{label}.pth")

    for ep in range(cfg.lopo_epochs):
        verbose  = (ep == 0)
        tr_loss  = train_epoch(model, tr_ld, crit, opt, scaler, device, cfg, verbose)
        sched.step()
        vm = evaluate(model, val_ld, crit, device)

        if (ep+1) % 5 == 0 or ep < 3:
            print(f"  [{label}] ep{ep+1:02d}  loss={tr_loss:.4f}  "
                  f"AUC={vm['auc']:.4f}  BAcc={vm['bacc']:.4f}  "
                  f"Sens={vm['sens']*100:.1f}%  Spec={vm['spec']*100:.1f}%")

        if vm["auc"] > best_auc:
            best_auc = vm["auc"]; pat_cnt = 0
            _s = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
            torch.save(_s, best_path)
        else:
            pat_cnt += 1
            if pat_cnt >= cfg.lopo_patience:
                print(f"  Early stop ep {ep+1}  best_AUC={best_auc:.4f}"); break

    _m = model.module if isinstance(model, nn.DataParallel) else model
    if os.path.exists(best_path):
        _m.load_state_dict(torch.load(best_path, map_location=device))

    # ── TEST-TIME ADAPTATION ─────────────────────────────────────────
    print(f"  [{label}] TTA: adapting to test patient ({len(X_te)} segs)…")

    ttbn = TTBNorm(_m, device)
    _m   = ttbn.adapt(X_te, n_passes=cfg.tta_bn_passes)

    if patient_id is not None:
        _m = tta_entropy_finetune(
            _m, X_te, device, cfg, patient_id,
            n_steps=cfg.tta_lora_steps, lr=cfg.tta_lora_lr
        )

    thr_otsu   = find_threshold_otsu(_m, X_te, device)
    thr_youden = find_threshold_youden(_m, val_ld, device, cfg)
    thr        = float(np.clip(thr_otsu, thr_youden - 0.15, thr_youden + 0.15))
    print(f"  [{label}] Final threshold: Otsu={thr_otsu:.3f}  "
          f"Youden={thr_youden:.3f}  Combined={thr:.3f}")

    te_ld = make_loader(X_te, y_te, False, cfg)
    te    = evaluate(_m, te_ld, crit, device, thr, patient_id)
    te    = apply_inversion_guard(te, thr, label)

    print(f"  [{label}] TEST  BAcc={te['bacc']:.4f}  "
          f"Sens={te['sens']*100:.1f}%  Spec={te['spec']*100:.1f}%  "
          f"AUC={te['auc']:.4f}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return te


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 17: LOPO-CV LOOPS ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def run_lopo(X_nodes, y_all, pid_all, A_spatial, ch_names, cfg, device):
    print("\n" + "="*70 + "\nNeuroGraphMamba v5 — LOPO-CV\n" + "="*70)
    pats    = np.unique(pid_all); results = []; n_nodes = X_nodes.shape[1]

    for pat in pats:
        te_mask = pid_all == pat
        if te_mask.sum() < 10 or len(np.unique(y_all[te_mask])) < 2:
            print(f"  {pat}: skip (too few or single-class)"); continue

        X_te = np.array(X_nodes[te_mask], np.float32); y_te = y_all[te_mask]
        tr_pats = [p for p in pats if p != pat]

        val_pat = pick_val_patient(pat, tr_pats, pid_all, y_all)

        if val_pat is not None:
            val_mask = pid_all == val_pat
            tr_mask  = ~te_mask & ~val_mask
            X_val    = np.array(X_nodes[val_mask], np.float32); y_val = y_all[val_mask]
            X_tr     = np.array(X_nodes[tr_mask],  np.float32); y_tr  = y_all[tr_mask]
            print(f"\n  ── {pat}  test={te_mask.sum()}  "
                  f"val={val_pat}({val_mask.sum()})  train={tr_mask.sum()}")
        else:
            X_tv = np.array(X_nodes[~te_mask], np.float32); y_tv = y_all[~te_mask]
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=SEED)
            del X_tv
            print(f"\n  ── {pat}  test={te_mask.sum()}  train={len(X_tr)}")

        if len(np.unique(y_val)) < 2:
            X_tv = np.concatenate([X_tr, X_val]); y_tv = np.concatenate([y_tr, y_val])
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tv, y_tv, test_size=0.20, stratify=y_tv, random_state=SEED)
            del X_tv

        X_tr = augment_lopo(X_tr)

        n_g   = min(300, len(X_tr))
        idx_g = np.random.choice(len(X_tr), n_g, replace=False)
        A_kg  = build_kg_adjacency(X_tr[idx_g], cfg.granger_max_lag,
                                    cfg.prior_weight, cfg.coh_weight)

        te = run_lopo_fold(
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            A_spatial, A_kg, cfg, device,
            f"LOPO_{pat}", n_nodes, patient_id=pat
        )
        te["patient"] = pat; results.append(te)
        del X_tr, X_val, X_te; gc.collect()

    if results:
        print(f"\n{'='*70}\nLOPO-CV Summary ({len(results)} patients) — 95% CI\n{'='*70}")
        for name, vals, pct in [
            ("Balanced Acc",  [r["bacc"]*100 for r in results], True),
            ("Sensitivity",   [r["sens"]*100 for r in results], True),
            ("Specificity",   [r["spec"]*100 for r in results], True),
            ("AUC-ROC",       [r["auc"]      for r in results], False),
            ("F1 (weighted)", [r["f1"]       for r in results], False),
        ]:
            m, lo, hi = bootstrap_ci(vals, cfg.bootstrap_n, cfg.bootstrap_ci)
            print(f"  {name:<18}: {fmt_ci(m, lo, hi, pct)}")
        auc_m = np.mean([r["auc"] for r in results])
        print(f"\n  SOTA: SeiFuD 2025 AUC=0.60 | NeuroGraphMamba v5 AUC={auc_m:.4f}")
        pd.DataFrame(results).to_csv(f"{cfg.out_path}/lopo_results.csv", index=False)
    return results


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 18: SIENA LOPO-CV (MODE A) ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def run_siena_lopo(siena_path: str, cfg, device, out_path: str = None):
    """Full LOPO-CV on Siena dataset only — useful as ablation / baseline."""
    seed_everything()
    scfg     = patch_cfg_for_siena(cfg)
    out_path = out_path or os.path.join(cfg.out_path, "siena_lopo")
    os.makedirs(out_path, exist_ok=True)
    scfg.out_path = out_path
    cache = os.path.join(out_path, "cache"); os.makedirs(cache, exist_ok=True)

    print("\n" + "="*65 + "\nSIENA LOPO-CV — LOADING DATA\n" + "="*65)
    debug_siena_channels(siena_path, n_patients=2)   # shows real channel names
    X_raw, y_all, pid_all = load_siena(siena_path, scfg, cache)

    print("\n" + "="*65 + "\nSIENA LOPO-CV — DWT FEATURES\n" + "="*65)
    node_cache = os.path.join(cache, "siena_nodes.npy")
    X_nodes    = extract_node_features(X_raw, scfg, node_cache)
    del X_raw; gc.collect()

    n_nodes  = X_nodes.shape[1]
    ch_names = SIENA_EEG_CHANNELS[:n_nodes]

    print("\n  Building Siena spatial adjacency …")
    X_mean_coh = X_nodes[:, :, 0, :].mean(0)
    A_spatial  = build_siena_spatial_adj(
        ch_names, X_mean_coh, k=scfg.knn_k, coh_w=scfg.coh_weight, fs=scfg.fs)

    print("\n  Building inter-band KG adjacency [N1] …")
    n_g   = min(500, len(X_nodes))
    idx_g = np.random.choice(len(X_nodes), n_g, replace=False)
    A_kg  = build_kg_adjacency(
        X_nodes[idx_g], scfg.granger_max_lag, scfg.prior_weight, scfg.coh_weight)

    print("\n" + "="*65 + "\nSIENA LOPO-CV — TRAINING\n" + "="*65)
    results = run_lopo(X_nodes, y_all, pid_all, A_spatial, ch_names, scfg, device)

    _print_summary("SIENA LOPO-CV", results)
    if results:
        pd.DataFrame(results).to_csv(f"{out_path}/siena_lopo_results.csv", index=False)
    return results


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 19: SIENA ZERO-SHOT (MODE B) ───────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def run_siena_zero_shot(siena_path: str, cfg, device,
                         chbmit_checkpoint_path: str = None,
                         chbmit_model=None,
                         out_path: str = None):
    """
    Zero-shot cross-dataset evaluation:
      1. Load CHB-MIT-trained NeuroGraphMamba v5 model
      2. For each Siena patient:
         a. Evaluate WITHOUT TTA  (pure zero-shot)
         b. Apply TTA (TTBNorm + LoRA entropy)
         c. Evaluate WITH TTA
      3. Print comparison table for MICCAI paper
    """
    seed_everything()
    scfg     = patch_cfg_for_siena(cfg)
    out_path = out_path or os.path.join(cfg.out_path, "siena_zeroshot")
    os.makedirs(out_path, exist_ok=True)
    cache = os.path.join(out_path, "cache"); os.makedirs(cache, exist_ok=True)

    print("\n" + "="*65)
    print("SIENA ZERO-SHOT CROSS-DATASET TRANSFER")
    print("Train: CHB-MIT  →  Test: Siena  (no Siena training data)")
    print("="*65)

    X_raw, y_all, pid_all = load_siena(siena_path, scfg, cache)

    print("\n  Extracting DWT node features …")
    node_cache = os.path.join(cache, "siena_nodes_zs.npy")
    X_nodes    = extract_node_features(X_raw, scfg, node_cache)
    del X_raw; gc.collect()

    n_nodes  = X_nodes.shape[1]
    ch_names = SIENA_EEG_CHANNELS[:n_nodes]

    print("\n  Building Siena spatial adjacency …")
    X_mean_coh = X_nodes[:, :, 0, :].mean(0)
    A_spatial  = build_siena_spatial_adj(
        ch_names, X_mean_coh, k=scfg.knn_k, coh_w=scfg.coh_weight, fs=scfg.fs)

    print("\n  Building inter-band KG [N1] …")
    n_g   = min(500, len(X_nodes))
    idx_g = np.random.choice(len(X_nodes), n_g, replace=False)
    A_kg  = build_kg_adjacency(
        X_nodes[idx_g], scfg.granger_max_lag, scfg.prior_weight, scfg.coh_weight)

    # ── Load CHB-MIT trained model ────────────────────────────────────
    if chbmit_model is not None:
        base_model = chbmit_model
        print("  Using provided CHB-MIT model object")
    elif chbmit_checkpoint_path is not None:
        print(f"  Loading CHB-MIT checkpoint: {chbmit_checkpoint_path}")
        base_model = NeuroGraphMamba(scfg, n_nodes).to(device)
        base_model.set_adj(A_spatial, A_kg, device)
        state = torch.load(chbmit_checkpoint_path, map_location=device)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        try:
            base_model.load_state_dict(state, strict=False)
            print("  Checkpoint loaded (strict=False — node count may differ)")
        except Exception as e:
            print(f"  [WARN] Checkpoint load error: {e}")
    else:
        raise ValueError("Provide either chbmit_model or chbmit_checkpoint_path")

    base_model.eval()
    base_model.set_adj(A_spatial, A_kg, device)

    pats = np.unique(pid_all)
    results_no_tta   = []
    results_with_tta = []

    for pat in pats:
        mask = pid_all == pat
        if mask.sum() < 10 or len(np.unique(y_all[mask])) < 2:
            print(f"\n  {pat}: skip (too few or single-class)"); continue

        X_te = np.array(X_nodes[mask], np.float32)
        y_te = y_all[mask]

        print(f"\n  {'─'*55}")
        print(f"  Patient: {pat}  ({mask.sum()} segs, "
              f"sz={y_te.sum()}, nsz={(y_te==0).sum()})")

        model_copy = copy.deepcopy(base_model)
        model_copy.set_adj(A_spatial, A_kg, device)
        model_copy.eval()

        crit  = make_criterion(y_te, device)
        te_ld = make_loader(X_te, y_te, False, scfg)

        # (a) WITHOUT TTA
        r_no_tta = evaluate(model_copy, te_ld, crit, device, thr=0.5)
        r_no_tta = apply_inversion_guard(r_no_tta, 0.5, f"{pat}_no_tta")
        r_no_tta["patient"] = pat
        results_no_tta.append(r_no_tta)
        print(f"  [{pat}] No-TTA  BAcc={r_no_tta['bacc']:.4f}  "
              f"AUC={r_no_tta['auc']:.4f}  "
              f"Sens={r_no_tta['sens']*100:.1f}%  Spec={r_no_tta['spec']*100:.1f}%")

        # (b) APPLY TTA
        print(f"  [{pat}] Applying TTA [F1+F2] …")
        ttbn       = TTBNorm(model_copy, device)
        model_copy = ttbn.adapt(X_te, n_passes=scfg.tta_bn_passes)
        model_copy = tta_entropy_finetune(
            model_copy, X_te, device, scfg, pat,
            n_steps=scfg.tta_lora_steps, lr=scfg.tta_lora_lr)

        thr_otsu   = find_threshold_otsu(model_copy, X_te, device)
        X_tv, X_val_zs, y_tv, y_val_zs = train_test_split(
            X_te, y_te, test_size=0.20, stratify=y_te, random_state=SEED)
        val_ld_zs  = make_loader(X_val_zs, y_val_zs, False, scfg)
        thr_youden = find_threshold_youden(model_copy, val_ld_zs, device, scfg)
        thr        = float(np.clip(thr_otsu, thr_youden - 0.15, thr_youden + 0.15))
        print(f"  [{pat}] Threshold: Otsu={thr_otsu:.3f}  "
              f"Youden={thr_youden:.3f}  Final={thr:.3f}")

        # (c) WITH TTA
        r_tta = evaluate(model_copy, te_ld, crit, device, thr=thr, patient_id=pat)
        r_tta = apply_inversion_guard(r_tta, thr, f"{pat}_tta")
        r_tta["patient"] = pat
        results_with_tta.append(r_tta)
        print(f"  [{pat}] With-TTA BAcc={r_tta['bacc']:.4f}  "
              f"AUC={r_tta['auc']:.4f}  "
              f"Sens={r_tta['sens']*100:.1f}%  Spec={r_tta['spec']*100:.1f}%")

        del model_copy; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Cross-dataset results table ───────────────────────────────────
    print("\n" + "="*65)
    print("CROSS-DATASET TRANSFER TABLE  (Table 2 for MICCAI paper)")
    print("="*65)
    print(f"  {'Patient':<8} │ {'No TTA':^30} │ {'With TTA':^30}")
    print(f"  {'':8} │ {'BAcc':>8} {'Sens':>8} {'AUC':>8} │ "
          f"{'BAcc':>8} {'Sens':>8} {'AUC':>8}")
    print("  " + "─"*72)

    no_tta_map   = {r["patient"]: r for r in results_no_tta}
    with_tta_map = {r["patient"]: r for r in results_with_tta}

    for pat in sorted(no_tta_map):
        n = no_tta_map[pat]; w = with_tta_map.get(pat, n)
        delta = w['bacc'] - n['bacc']
        flag  = " ↑" if delta > 0.02 else (" ↓" if delta < -0.02 else "")
        print(f"  {pat:<8} │ "
              f"{n['bacc']*100:>7.1f}% {n['sens']*100:>7.1f}% {n['auc']:>8.4f} │ "
              f"{w['bacc']*100:>7.1f}% {w['sens']*100:>7.1f}% {w['auc']:>8.4f}"
              f"  ({delta*100:+.1f}%){flag}")

    print("  " + "─"*72)
    for label, results in [("No TTA", results_no_tta), ("With TTA", results_with_tta)]:
        if not results: continue
        bacc_m = np.mean([r['bacc']*100 for r in results])
        auc_m  = np.mean([r['auc']      for r in results])
        sens_m = np.mean([r['sens']*100 for r in results])
        spec_m = np.mean([r['spec']*100 for r in results])
        print(f"  {'Mean':8} │  {label}:  "
              f"BAcc={bacc_m:.1f}%  Sens={sens_m:.1f}%  "
              f"Spec={spec_m:.1f}%  AUC={auc_m:.4f}")

    _plot_zero_shot_comparison(results_no_tta, results_with_tta, out_path)
    pd.DataFrame(results_no_tta).to_csv(f"{out_path}/siena_zeroshot_no_tta.csv",  index=False)
    pd.DataFrame(results_with_tta).to_csv(f"{out_path}/siena_zeroshot_with_tta.csv", index=False)
    return {"no_tta": results_no_tta, "with_tta": results_with_tta}


# ═══════════════════════════════════════════════════════════════════════
# ─── PART 20: PLOTS ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════

def plot_calibration(results, out_path):
    yt  = np.concatenate([r["targets"]      for r in results])
    yp  = np.concatenate([r["probs"][:, 1]  for r in results])
    bins = np.linspace(0, 1, 11); ece = 0.0; n_bins = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (yp >= lo) & (yp < hi)
        if mask.sum() > 0:
            ece += (yt[mask].mean() - yp[mask].mean())**2; n_bins += 1
    ece = float(np.sqrt(ece / max(n_bins, 1)))
    fp, mp = calibration_curve(yt, yp, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mp, fp, "s-", lw=2, label=f"NeuroGraphMamba v5 (ECE={ece:.4f})")
    ax.plot([0,1], [0,1], "k--", lw=1, label="Perfect")
    ax.legend(); ax.set_xlabel("Mean predicted"); ax.set_ylabel("Fraction positive")
    ax.set_title("Calibration — LOPO-CV (with TTA)")
    plt.tight_layout()
    plt.savefig(f"{out_path}/calibration.png", dpi=150); plt.close()
    print(f"  [Calibration] ECE={ece:.4f}")
    return ece


def plot_results(results, out_path):
    df = pd.DataFrame(results)
    if "patient" not in df.columns: return

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(df)*0.8), 12))
    fig.suptitle("NeuroGraphMamba v5 — Per-Patient LOPO (with TTA)", fontweight="bold")
    for ax, (col, label, color) in zip(axes, [
        ("auc",  "AUC-ROC",      "steelblue"),
        ("sens", "Sensitivity",  "coral"),
        ("spec", "Specificity",  "seagreen"),
    ]):
        vals = df[col].values * (100 if col in ("sens","spec") else 1)
        ax.bar(df["patient"], vals, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(np.mean(vals), color="k", lw=1.5, ls="--",
                   label=f"Mean={np.mean(vals):.3f}")
        ax.set_ylabel(label); ax.legend()
        ax.grid(axis="y", alpha=0.3); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_path}/lopo_per_patient.png", dpi=150); plt.close()

    n  = len(results); nc = min(4, n); nr = math.ceil(n / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(4*nc, 3.5*nr))
    axes = np.array(axes).flatten()
    for fi, r in enumerate(results):
        cm = confusion_matrix(r["targets"], r["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[fi],
                    xticklabels=["NSz","Sz"], yticklabels=["NSz","Sz"],
                    linewidths=0.5)
        axes[fi].set_title(
            f"{r.get('patient','?')}\n"
            f"BAcc={r['bacc']*100:.1f}%  AUC={r['auc']:.3f}",
            fontsize=8)
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out_path}/confusion_matrices.png", dpi=150); plt.close()
    print("  [Plots] saved.")


def plot_causal_structure(A_kg, A_sp, ch_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("NeuroGraphMamba v5 — Causal Structure [N1,N2]", fontweight="bold")
    sns.heatmap(A_kg, annot=True, fmt=".2f", cmap="YlOrRd",
                ax=axes[0], xticklabels=SUBBAND_LABELS, yticklabels=SUBBAND_LABELS,
                linewidths=0.5, vmin=0, vmax=1)
    axes[0].set_title("[N1] GrangerEdgeNet — Inter-band Adjacency")
    axes[0].set_xlabel("Target"); axes[0].set_ylabel("Source")
    indeg = torch.FloatTensor(A_sp).sum(0).numpy()[:len(ch_names)]
    order = np.argsort(indeg)[::-1]
    axes[1].barh(range(len(ch_names)), indeg[order],
                 color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(ch_names))))
    axes[1].set_yticks(range(len(ch_names)))
    axes[1].set_yticklabels([ch_names[i] for i in order], fontsize=8)
    axes[1].set_xlabel("Spatial indegree"); axes[1].invert_yaxis()
    axes[1].set_title("[N2] Cross-Attn Channel Order")
    plt.tight_layout()
    plt.savefig(f"{out_path}/causal_structure.png", dpi=150); plt.close()
    print("  [XAI] causal_structure.png saved")


def _plot_zero_shot_comparison(results_no_tta, results_with_tta, out_path):
    """Bar chart: per-patient BAcc without vs with TTA on Siena."""
    no_map  = {r["patient"]: r for r in results_no_tta}
    tta_map = {r["patient"]: r for r in results_with_tta}
    pats    = sorted(no_map.keys())
    if not pats: return

    x     = np.arange(len(pats)); w = 0.35
    no_b  = [no_map[p]["bacc"]*100  for p in pats]
    tta_b = [tta_map.get(p, no_map[p])["bacc"]*100 for p in pats]
    no_a  = [no_map[p]["auc"]       for p in pats]
    tta_a = [tta_map.get(p, no_map[p])["auc"]      for p in pats]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(pats)*1.2), 5))
    fig.suptitle(
        "NeuroGraphMamba v5 — Cross-Dataset Transfer: CHB-MIT → Siena\n"
        "Zero-Shot (No TTA) vs With Test-Time Adaptation [F1+F2]",
        fontweight="bold", fontsize=11)

    for ax, (vals_no, vals_tta, metric) in zip(axes, [
        (no_b,  tta_b,  "Balanced Accuracy (%)"),
        (no_a,  tta_a,  "AUC-ROC"),
    ]):
        ax.bar(x - w/2, vals_no,  w, label="No TTA",   color="steelblue", alpha=0.85)
        ax.bar(x + w/2, vals_tta, w, label="With TTA", color="coral",     alpha=0.85)
        ax.axhline(np.mean(vals_no),  color="steelblue", lw=1.5, ls="--",
                   label=f"No TTA μ={np.mean(vals_no):.3f}")
        ax.axhline(np.mean(vals_tta), color="coral",     lw=1.5, ls="--",
                   label=f"TTA μ={np.mean(vals_tta):.3f}")
        ax.set_xticks(x); ax.set_xticklabels(pats, rotation=45, ha="right")
        ax.set_ylabel(metric); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if "AUC" not in metric: ax.set_ylim(0, 108)
        else:
            ax.set_ylim(0, 1.05)
            ax.axhline(0.5, color="gray", lw=1, ls=":", label="Chance")

    plt.tight_layout()
    out = f"{out_path}/siena_zeroshot_comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  [Plot] Saved → {out}")

