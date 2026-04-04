"""
Base Sequence Toolkit — XEPV Analysis Framework for AI Agent Execution Traces

Classifies AI agent tool-call sequences into four base types:
  X (eXplore) — information gathering, browsing, searching
  E (Execute) — state-changing actions, writing, modifying
  P (Plan)    — pure reasoning / strategy formulation
  V (Verify)  — validation, testing, checking results

Provides analysis primitives: n-gram patterns, transition matrices,
positional effects, run-length statistics, and discriminative pattern mining.
"""

__version__ = "0.1.0"

from base_sequence_toolkit.core.classifier import (
    BaseType,
    ClassifierContext,
    StepClassification,
    classify_step,
    create_context,
)
from base_sequence_toolkit.core.analyzer import (
    BaseDistribution,
    TransitionMatrix,
    build_distribution,
    build_sequence_string,
    compute_ngrams,
    compute_transition_matrix,
    find_discriminative_patterns,
)

__all__ = [
    "BaseType",
    "ClassifierContext",
    "StepClassification",
    "classify_step",
    "create_context",
    "BaseDistribution",
    "TransitionMatrix",
    "build_distribution",
    "build_sequence_string",
    "compute_ngrams",
    "compute_transition_matrix",
    "find_discriminative_patterns",
]
