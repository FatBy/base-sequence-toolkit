"""
Base Sequence Toolkit — XEPV Analysis Framework for AI Agent Execution Traces

Classifies AI agent tool-call sequences into four base types:
  X (eXplore) — information gathering, browsing, searching
  E (Execute) — state-changing actions, writing, modifying
  P (Plan)    — pure reasoning / strategy formulation
  V (Verify)  — validation, testing, checking results

Provides analysis primitives: n-gram patterns, transition matrices,
positional effects, run-length statistics, and discriminative pattern mining.

Also includes the Governor — a three-layer adaptive intervention system:
  Layer 1: Online rule engine (7 rules, 8-dim features, 0ms latency)
  Layer 2: Statistical accumulator (bucket-based A/B tracking)
  Layer 3: Threshold self-adaptation (chi-squared significance testing)
"""

__version__ = "0.2.0"

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
from base_sequence_toolkit.core.governor import (
    BaseSequenceGovernor,
    GovernorSignal,
    GovernorStats,
    InterventionRecord,
    RuleThresholds,
    evaluate_sequence,
    extract_features,
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
    "BaseSequenceGovernor",
    "GovernorSignal",
    "GovernorStats",
    "InterventionRecord",
    "RuleThresholds",
    "evaluate_sequence",
    "extract_features",
]
