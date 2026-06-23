# Reproducibility notes

## Dataset access
- **CHB-MIT Scalp EEG Database**: open access via PhysioNet (ODC-BY 1.0).
  https://physionet.org/content/chbmit/1.0.0/
- **Siena Scalp EEG Database**: requires PhysioNet credentialed access.
  https://physionet.org/content/siena-scalp-eeg/1.0.0/

Neither dataset should be committed to this repository.

## Known data quirks handled by the loaders
These are real properties of the raw files, not bugs — documented here so
a future maintainer (or a reviewer trying to reproduce results) doesn't
"fix" them away:

- Siena annotation times are wall-clock (`HH:MM:SS`), not seconds-from-file-
  start; converting requires the per-file registration start time, which is
  not always present and is approximated for a few files (e.g. `PN01-1.edf`).
- Some Siena filenames use the letter `O` instead of digit `0`
  (`PNO6-1.edf`), which is preserved verbatim rather than "corrected," since
  it must match the actual file on disk.
- `PN00-3.edf` seizure 3's annotated end time exceeds the file's actual
  duration — this is a known annotation error in the source data and is
  clamped to file length.
- Siena EDF headers include non-EEG channels (`EKG`, `SPO2`, `HR`, `MK`)
  and a channel literally labelled `"1"` that must be excluded from the
  29-channel montage.
- `PN08` has no annotated seizures and is automatically skipped — this is
  why the paper reports 14 Siena patients total but 13 with seizures.

## Before resubmitting / re-running
If you change any annotation-parsing logic, re-run
`scripts/run_siena_lopo.py` and confirm the per-patient seizure/non-seizure
counts printed to stdout match what's reported in the paper's results
tables before trusting new numbers — this pipeline has previously had
silent regressions where a parsing change caused valid patients to return
zero seizures and get skipped rather than raising an error.

## Expected runtime (rough, single GPU)
- CHB-MIT LOPO-CV (24 folds): several hours, dominated by EDF loading +
  per-fold training + TTA.
- Siena LOPO-CV (13 folds): faster, smaller dataset.
- Siena zero-shot: fastest — no training, just TTA + evaluation per patient.

## Ablations worth running before final submission
(see reviewer notes — these are currently missing from the paper)
- `prior_weight = 0.0` and `kg_consistency_lam = 0.0` together, to report
  what the causal graph looks like *without* the neuroanatomical prior and
  ordering loss, supporting (or correcting) the "emergent structure" claim.
- TTA ablation: report AUC with TTBNorm only, LoRA-entropy only, both, and
  neither, per dataset.
