"""
neurographmamba/core.py
========================
PASTE YOUR FULL PIPELINE SCRIPT HERE (the one currently named something
like `neurograpghmamba_v5_unified.py`), with exactly two edits:

EDIT 1 — remove the Config class, use the shared one instead
--------------------------------------------------------------
Delete this whole block from your script:

    class Config:
        device = torch.device(...)
        ...
    cfg = Config()
    os.makedirs(cfg.out_path, exist_ok=True)

And add this near the top instead (after your other imports):

    from configs.default import cfg

`configs/default.py` already has every field your Config class had —
just edit the paths there instead of in this file.

EDIT 2 — remove the RUN_MODE / main() driver block at the bottom
--------------------------------------------------------------
Delete:

    RUN_MODE = "both"
    ...
    def main():
        ...
    if __name__ == "__main__":
        main()

The three scripts in scripts/ replace this — each one imports the specific
function it needs (run_lopo, run_siena_lopo, run_siena_zero_shot) and
calls it directly, so you can run CHB-MIT, Siena LOPO, or Siena zero-shot
independently without editing a RUN_MODE string.

Everything else — all classes, functions, the PART 1-21 structure — stays
exactly as-is. This module is intentionally one file: it's internally
cross-referencing (the model classes call the graph builders, training
calls the TTA functions, etc.) and a deeper modular split should only be
done once you've re-run the full pipeline end-to-end against real data
to confirm nothing broke in the split.
"""

raise NotImplementedError(
    "Paste your pipeline script here (see the module docstring above for "
    "the two edits needed) before importing neurographmamba."
)
