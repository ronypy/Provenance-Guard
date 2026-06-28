"""Provenance Guard detection pipeline.

Public surface:

* ``run_pipeline(text)``  -> dict with every signal score, the combined score,
                             confidence, attribution, and the reader-facing label.
"""

from .pipeline import run_pipeline

__all__ = ["run_pipeline"]
