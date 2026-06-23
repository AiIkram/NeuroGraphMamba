"""
neurographmamba
================
Public package init. Re-exports the entry points scripts/ actually use,
so callers can do `from neurographmamba import run_lopo` etc. without
caring about internal layout.

NOTE: populate neurographmamba/core.py with your pipeline code (see the
docstring at the top of core.py for the two small edits needed) before
this import will succeed.
"""
from .core import (  # noqa: F401
    seed_everything,
    load_chbmit,
    load_siena,
    load_all_siena_annotations,
    extract_node_features,
    build_kg_adjacency,
    build_spatial_adjacency,
    build_siena_spatial_adj,
    NeuroGraphMamba,
    run_lopo,
    run_siena_lopo,
    run_siena_zero_shot,
    debug_siena_channels,
)

__all__ = [
    "seed_everything",
    "load_chbmit",
    "load_siena",
    "load_all_siena_annotations",
    "extract_node_features",
    "build_kg_adjacency",
    "build_spatial_adjacency",
    "build_siena_spatial_adj",
    "NeuroGraphMamba",
    "run_lopo",
    "run_siena_lopo",
    "run_siena_zero_shot",
    "debug_siena_channels",
]
