"""
Baseline Methods for Comparison
===============================

Provides baseline implementations to compare against LLM evolution:
- StaticBaseline: No evolution, uses fixed EMA strategy
- RandomMutationBaseline: Random parameter mutations
- SingleShotLLMBaseline: LLM generates once, no iteration
"""

from .static import StaticBaseline
from .random_mutation import RandomMutationBaseline
from .single_shot import SingleShotLLMBaseline

__all__ = [
    "StaticBaseline",
    "RandomMutationBaseline",
    "SingleShotLLMBaseline",
]
