"""
DunCrew ExecTrace Adapter

Reads DunCrew execution trace files (JSONL format) and converts them into
XEPV base sequences for analysis.

DunCrew stores execution traces as JSONL files in the memory/exec_traces/
directory, with fields like:
  - baseSequence: "X-E-P-V-E" (dash-separated)
  - baseDistribution: {E: 3, P: 1, V: 1, X: 1}
  - success: bool
  - task: str
  - tools: [{name, args, status, baseType}, ...]

This adapter can work in two modes:
  1. **Pre-classified**: Read existing baseSequence from trace files
  2. **Re-classify**: Re-run the classifier on raw tool calls

Usage:
    from base_sequence_toolkit.adapters.duncrew import load_traces, load_and_reclassify

    # Mode 1: Use existing classifications
    results = load_traces("path/to/exec_traces/")

    # Mode 2: Re-classify from raw tool calls
    results = load_and_reclassify("path/to/exec_traces/")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from base_sequence_toolkit.core.classifier import (
    BaseType,
    ClassifierContext,
    StepClassification,
    classify_step,
    create_context,
)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class DunCrewTraceResult:
    """XEPV classification result for a single DunCrew execution trace."""
    trace_id: str
    task: str
    resolved: bool
    base_sequence: str  # compact form, e.g. "XEPVE"
    step_count: int
    base_counts: dict[str, int] = field(default_factory=dict)
    model: str = ""
    timestamp: float = 0.0

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
            "trace_id": self.trace_id,
            "task": self.task,
            "resolved": self.resolved,
            "base_sequence": self.base_sequence,
            "step_count": self.step_count,
            "x_ratio": round(self.x_ratio, 3),
            "e_ratio": round(self.e_ratio, 3),
            "p_ratio": round(self.p_ratio, 3),
            "v_ratio": round(self.v_ratio, 3),
            "model": self.model,
        }


# ── File Reading ─────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Read a JSONL file, skipping corrupted lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _iter_trace_dir(trace_dir: str | Path) -> Iterator[dict]:
    """Iterate over all JSONL trace files in a directory."""
    trace_path = Path(trace_dir)
    for file in sorted(trace_path.glob("*.jsonl")):
        yield from _iter_jsonl(file)


# ── Mode 1: Pre-classified Traces ───────────────────────────────────────────

def _parse_pre_classified(trace: dict) -> DunCrewTraceResult | None:
    """Parse a trace that already has baseSequence field."""
    seq_field = trace.get("baseSequence", "")
    success = trace.get("success")

    if not seq_field or success is None:
        return None

    # Handle both "X-E-P-V" and list formats
    if isinstance(seq_field, str):
        bases = seq_field.replace("-", "")
    elif isinstance(seq_field, list):
        bases = "".join(seq_field)
    else:
        return None

    # Use pre-computed distribution if available
    dist = trace.get("baseDistribution", {})
    if not dist:
        from collections import Counter
        dist = dict(Counter(bases))

    return DunCrewTraceResult(
        trace_id=trace.get("id", ""),
        task=trace.get("task", ""),
        resolved=success,
        base_sequence=bases,
        step_count=len(bases),
        base_counts=dist,
        model=trace.get("llmModel", ""),
        timestamp=trace.get("timestamp", 0.0),
    )


def load_traces(
    trace_dir: str | Path,
    max_samples: int | None = None,
) -> list[DunCrewTraceResult]:
    """Load pre-classified DunCrew execution traces.

    Reads the existing baseSequence field from trace files.
    This is the fastest mode — no re-classification is performed.

    Args:
        trace_dir: Path to the exec_traces directory.
        max_samples: Maximum number of traces to load.

    Returns:
        List of DunCrewTraceResult.
    """
    results: list[DunCrewTraceResult] = []
    for trace in _iter_trace_dir(trace_dir):
        result = _parse_pre_classified(trace)
        if result:
            results.append(result)
            if max_samples and len(results) >= max_samples:
                break
    return results


# ── Mode 2: Re-classify from Raw Tool Calls ─────────────────────────────────

def _reclassify_trace(trace: dict) -> DunCrewTraceResult | None:
    """Re-classify a trace from its raw tool call data."""
    tools = trace.get("tools", [])
    success = trace.get("success")

    if not tools or success is None:
        return None

    ctx = create_context()
    bases: list[str] = []

    for i, tool_call in enumerate(tools):
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        status = tool_call.get("status", "success")

        # Check for pre-existing P classification from LLM
        pre_base = None
        if tool_call.get("baseType") == "P":
            pre_base = BaseType.P

        step = StepClassification(
            tool_name=name,
            args=args,
            status=status,
            shell_command=str(args.get("command", args.get("cmd", ""))) if name == "runCmd" else None,
            pre_assigned_base=pre_base,
            order=i,
        )

        base = classify_step(step, ctx)
        bases.append(str(base))

    if not bases:
        return None

    from collections import Counter
    seq = "".join(bases)
    counts = dict(Counter(bases))

    return DunCrewTraceResult(
        trace_id=trace.get("id", ""),
        task=trace.get("task", ""),
        resolved=success,
        base_sequence=seq,
        step_count=len(bases),
        base_counts=counts,
        model=trace.get("llmModel", ""),
        timestamp=trace.get("timestamp", 0.0),
    )


def load_and_reclassify(
    trace_dir: str | Path,
    max_samples: int | None = None,
) -> list[DunCrewTraceResult]:
    """Load and re-classify DunCrew execution traces from raw tool calls.

    This re-runs the XEPV classifier on each tool call, which can produce
    different results if the classifier logic has been updated.

    Args:
        trace_dir: Path to the exec_traces directory.
        max_samples: Maximum number of traces to load.

    Returns:
        List of DunCrewTraceResult.
    """
    results: list[DunCrewTraceResult] = []
    for trace in _iter_trace_dir(trace_dir):
        result = _reclassify_trace(trace)
        if result:
            results.append(result)
            if max_samples and len(results) >= max_samples:
                break
    return results
