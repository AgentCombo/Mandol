"""Legacy multi-dimensional semantic graph system.

Retained for backwards compatibility with older pipeline configurations.
Prefer the UnifiedFactPipeline under mandol.application.pipeline for new
development.
"""

from .mdsg_pipeline import run_mdsg_pipeline as run_mdsg_pipeline
