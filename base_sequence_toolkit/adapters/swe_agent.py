"""
SWE-agent Trajectory Adapter

Maps SWE-agent trajectories (from nebius/SWE-agent-trajectories or similar
datasets) into XEPV base sequences.

SWE-agent forces an action every step via code blocks in AI messages.
This means P bases are expected to be rare — this is itself a structural
finding (see paper section on "forced-action bias").

Usage:
    from base_sequence_toolkit.adapters.swe_agent import (
        load_from_huggingface, classify_trajectory, batch_classify
    )

    # From HuggingFace dataset
    results = load_from_huggingface(max_records=500)

    # Or classify a single trajectory dict
    result = classify_trajectory(trajectory_record)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

from base_sequence_toolkit.core.classifier import BaseType


# ── SWE-agent Specific Classification Rules ──────────────────────────────────

EXPLORE_COMMANDS: set[str] = {
    "search_dir", "search_file", "find_file", "find", "grep",
    "ls", "open", "goto", "scroll_down", "scroll_up", "pwd",
    "cat", "head", "tail", "less", "more", "wc", "tree", "file",
    "which", "type", "echo",
}

EXECUTE_COMMANDS: set[str] = {
    "edit", "create", "rm", "mv", "cp", "mkdir", "chmod", "chown",
    "sed", "awk", "patch", "touch",
    "pip", "npm", "git",
    "cd",
}

VERIFY_COMMANDS: set[str] = {
    "pytest", "unittest", "submit",
}

VERIFY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"pytest",
        r"python\s+.*test",
        r"python\s+-m\s+unittest",
        r"python\s+-m\s+pytest",
        r"python\s+.*reproduce",
        r"npm\s+test",
        r"make\s+test",
        r"tox\b",
        r"nosetests",
        r"^submit$",
    ]
]

PYTHON_VERIFY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"python\s+.*test",
        r"python\s+.*reproduce",
        r"python\s+-m\s+(unittest|pytest|nose)",
        r"python\s+-c\s+.*assert",
    ]
]

PYTHON_EXPLORE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"python\s+-c\s+.*print",
        r"python\s+-c\s+.*import",
        r"python\s+--version",
    ]
]


# ── Command Extraction ───────────────────────────────────────────────────────

def extract_command(ai_text: str) -> str | None:
    """Extract the shell command from an AI message's code block.

    SWE-agent wraps commands in ``` code blocks.
    Returns the last code block content, or None if no code block found.
    """
    if not ai_text:
        return None
    matches = re.findall(r"```\n?(.*?)\n?```", ai_text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


# ── Step Classification ──────────────────────────────────────────────────────

def classify_swe_step(ai_text: str) -> BaseType:
    """Classify a single SWE-agent step into an XEPV base.

    Priority:
      1. No code block → P (pure planning/thinking)
      2. Full-line verify patterns → V
      3. Verb-based classification (VERIFY > EXPLORE > EXECUTE)
      4. Special python handling
      5. Default → E
    """
    cmd = extract_command(ai_text)

    # No code block = pure reasoning
    if cmd is None or not cmd.strip():
        return BaseType.P

    first_line = cmd.strip().split("\n")[0].strip()
    parts = first_line.split()
    if not parts:
        return BaseType.P

    verb = parts[0].lower()

    # Priority 1: Full-line verify patterns
    if verb == "submit":
        return BaseType.V
    for pattern in VERIFY_PATTERNS:
        if pattern.search(first_line):
            return BaseType.V

    # Priority 2: Verb-based
    if verb in VERIFY_COMMANDS:
        return BaseType.V
    if verb in EXPLORE_COMMANDS:
        return BaseType.X
    if verb in EXECUTE_COMMANDS:
        return BaseType.E

    # Priority 3: Special python handling
    if verb in ("python", "python3"):
        for p in PYTHON_VERIFY_PATTERNS:
            if p.search(first_line):
                return BaseType.V
        for p in PYTHON_EXPLORE_PATTERNS:
            if p.search(first_line):
                return BaseType.X
        return BaseType.E

    # Priority 4: Executable paths
    if verb.startswith("./") or verb.startswith("/"):
        return BaseType.E

    # Default
    return BaseType.E


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SWETrajectoryResult:
    """XEPV classification result for a single SWE-agent trajectory."""
    instance_id: str
    model_name: str
    resolved: bool
    base_sequence: str  # compact form, e.g. "XEXEVXE"
    step_count: int
    base_counts: dict[str, int] = field(default_factory=dict)
    exit_status: str = ""

    @property
    def x_ratio(self) -> float:
        return self.base_counts.get("X", 0) / max(self.step_count, 1)

    @property
    def e_ratio(self) -> float:
        return self.base_counts.get("E", 0) / max(self.step_count, 1)

    @property
    def p_ratio(self) -> float:
        return self.base_counts.get("P", 0) / max(self.step_count, 1)

    @property
    def v_ratio(self) -> float:
        return self.base_counts.get("V", 0) / max(self.step_count, 1)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model_name": self.model_name,
            "resolved": self.resolved,
            "base_sequence": self.base_sequence,
            "step_count": self.step_count,
            "x_ratio": round(self.x_ratio, 3),
            "e_ratio": round(self.e_ratio, 3),
            "p_ratio": round(self.p_ratio, 3),
            "v_ratio": round(self.v_ratio, 3),
            "exit_status": self.exit_status,
        }


# ── Trajectory Classification ────────────────────────────────────────────────

def classify_trajectory(record: dict) -> SWETrajectoryResult:
    """Classify a single SWE-agent trajectory record into XEPV base sequence.

    Args:
        record: A trajectory record with keys:
            - trajectory: list of {role, text} items
            - instance_id: str
            - model_name: str
            - target: bool (resolved or not)
            - exit_status: str (optional)

    Returns:
        SWETrajectoryResult with the classified sequence.
    """
    traj = record["trajectory"]
    bases: list[str] = []

    for item in traj:
        if item["role"] == "ai" and item.get("text"):
            base = classify_swe_step(item["text"])
            bases.append(str(base))

    seq = "".join(bases)
    counts = dict(Counter(bases))

    return SWETrajectoryResult(
        instance_id=record["instance_id"],
        model_name=record["model_name"],
        resolved=record["target"],
        base_sequence=seq,
        step_count=len(bases),
        base_counts=counts,
        exit_status=record.get("exit_status", ""),
    )


def batch_classify(records: Iterator[dict], max_records: int | None = None) -> list[SWETrajectoryResult]:
    """Classify a batch of SWE-agent trajectory records.

    Args:
        records: Iterator of trajectory records.
        max_records: Maximum number of records to process (None = all).

    Returns:
        List of SWETrajectoryResult.
    """
    results: list[SWETrajectoryResult] = []
    for i, rec in enumerate(records):
        if max_records is not None and i >= max_records:
            break
        results.append(classify_trajectory(rec))
    return results


# ── HuggingFace Integration ─────────────────────────────────────────────────

def load_from_huggingface(
    dataset_name: str = "nebius/SWE-agent-trajectories",
    split: str = "train",
    max_records: int = 1000,
) -> list[SWETrajectoryResult]:
    """Load and classify SWE-agent trajectories from HuggingFace.

    Requires: pip install datasets

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use.
        max_records: Maximum number of records to process.

    Returns:
        List of SWETrajectoryResult.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace integration. "
            "Install it with: pip install base-sequence-toolkit[swe-agent]"
        )

    ds = load_dataset(dataset_name, split=split, streaming=True)
    return batch_classify(iter(ds), max_records=max_records)
