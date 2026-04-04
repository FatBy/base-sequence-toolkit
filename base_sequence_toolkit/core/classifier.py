"""
Generic XEPV Base Type Classifier

Classifies AI agent actions (tool calls, shell commands, etc.) into four base types:

  X (eXplore) — Information gathering: search, browse, read unknown resources
  E (Execute) — State-changing actions: write files, run commands, modify state
  P (Plan)    — Pure reasoning / strategy (must be assigned by LLM or adapter)
  V (Verify)  — Validation: run tests, check results, compare outputs

This module provides a *stateful* classifier: classification depends on the
history of previous actions (e.g., reading a file you just wrote is V, not X).

Adapter authors implement `extract_step(raw_data) -> StepClassification` and
call `classify_step()` to get the base type.

NOTE: P (Plan) cannot be reliably inferred from tool calls alone.
      It must be assigned by the LLM itself or by adapter-specific heuristics.
      This classifier handles E/V/X; P is passed through when pre-assigned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


# ── Base Type ────────────────────────────────────────────────────────────────

class BaseType(str, Enum):
    """The four XEPV base types."""
    X = "X"  # eXplore
    E = "E"  # Execute
    P = "P"  # Plan
    V = "V"  # Verify

    def __str__(self) -> str:
        return self.value


# ── Tool Categories ──────────────────────────────────────────────────────────

READ_TOOLS: set[str] = {
    "readFile", "read_file", "listDir", "list_dir",
    "searchText", "search_text", "searchFiles", "search_files",
    "readMultipleFiles", "read_multiple_files",
    # SWE-agent specific
    "open", "scroll_down", "scroll_up", "goto",
    "search_dir", "search_file", "find_file",
}

WRITE_TOOLS: set[str] = {
    "writeFile", "write_file", "appendFile", "append_file",
    "deleteFile", "delete_file", "renameFile", "rename_file",
    # SWE-agent specific
    "edit", "create",
}

EXPLORE_TOOLS: set[str] = {
    "webSearch", "web_search", "webFetch", "web_fetch",
}

VERIFY_CMD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"tsc\b.*--noEmit", re.I),
    re.compile(r"npm\s+(test|run\s+test|run\s+lint|run\s+build)", re.I),
    re.compile(r"pytest|jest|vitest|mocha", re.I),
    re.compile(r"eslint|prettier.*--check", re.I),
    re.compile(r"cargo\s+(check|test|clippy)", re.I),
    re.compile(r"go\s+(test|vet)", re.I),
    re.compile(r"python\s+-m\s+(unittest|pytest)", re.I),
    re.compile(r"make\s+test", re.I),
    re.compile(r"tox\b", re.I),
    re.compile(r"nosetests", re.I),
]

EXPLORE_CMD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(ls|dir|tree|find)\b", re.I),
    re.compile(r"^(cat|head|tail|less|more)\b", re.I),
    re.compile(r"^(grep|rg|ag|ack)\b", re.I),
    re.compile(r"^git\s+(log|status|diff|show|branch)", re.I),
    re.compile(r"^(which|where|type|command\s+-v)\b", re.I),
    re.compile(r"^(echo\s+\$|env|printenv|set)\b", re.I),
    re.compile(r"^pwd\b", re.I),
    re.compile(r"^wc\b", re.I),
]

EXPLORE_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I) for p in [
        r"search", r"fetch", r"get", r"list", r"read",
        r"find", r"query", r"browse", r"scan", r"lookup",
        r"navigate", r"screenshot", r"inspect",
    ]
]


# ── Step Classification ──────────────────────────────────────────────────────

@dataclass
class StepClassification:
    """Input to the classifier — one agent step / tool call.

    Adapters fill this from their raw data format. At minimum, provide
    `tool_name`. If you can extract a resource path or shell command,
    populate those fields for better classification accuracy.

    If the adapter can determine a pre-assigned base type (e.g., the LLM
    labeled this step as P), set `pre_assigned_base`.
    """
    tool_name: str
    args: dict[str, object] = field(default_factory=dict)
    status: Literal["success", "error"] = "success"
    resource: str | None = None  # file path, URL, or query
    shell_command: str | None = None  # for runCmd-type tools
    pre_assigned_base: BaseType | None = None
    order: int = 0  # step index in the execution


# ── Classifier Context (stateful) ────────────────────────────────────────────

@dataclass
class ClassifierContext:
    """Tracks execution history for context-dependent classification.

    Create one per execution trace; feed steps sequentially.
    """
    successful_resources: set[str] = field(default_factory=set)
    recent_writes: list[tuple[str, int]] = field(default_factory=list)  # (resource, order)
    last_step: StepClassification | None = None
    last_base: BaseType | None = None

    def _record(self, step: StepClassification, base: BaseType) -> None:
        """Update context after classification."""
        resource = step.resource
        if step.status == "success" and resource:
            self.successful_resources.add(resource)
        if step.tool_name in WRITE_TOOLS and step.status == "success" and resource:
            self.recent_writes.append((resource, step.order))
            if len(self.recent_writes) > 10:
                self.recent_writes.pop(0)
        self.last_step = step
        self.last_base = base


def create_context() -> ClassifierContext:
    """Create a fresh classifier context."""
    return ClassifierContext()


# ── Path Normalization ───────────────────────────────────────────────────────

def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./").replace("//", "/")


def _extract_resource(step: StepClassification) -> str | None:
    """Best-effort resource extraction from step args."""
    if step.resource:
        return _normalize_path(step.resource) if "/" in step.resource or "\\" in step.resource else step.resource

    args = step.args
    for key in ("path", "filePath", "file_path", "file", "directory"):
        val = args.get(key)
        if isinstance(val, str):
            return _normalize_path(val)

    for key in ("url", "href"):
        val = args.get(key)
        if isinstance(val, str):
            return val

    for key in ("query", "q", "search_query"):
        val = args.get(key)
        if isinstance(val, str):
            return f"query:{val}"

    if step.shell_command:
        m = re.search(r"(?:^|\s)((?:\./|/|[a-zA-Z]:\\)[\w\-./\\]+)", step.shell_command)
        return _normalize_path(m.group(1)) if m else f"cmd:{step.shell_command[:50]}"

    return None


def _infer_intent_from_args(args: dict[str, object]) -> Literal["read", "write", "unknown"]:
    """Infer read/write intent from parameter names (fallback for unknown tools)."""
    has_content = any(args.get(k) for k in ("content", "body", "data", "text", "payload"))
    has_target = any(args.get(k) for k in ("url", "href", "query", "q", "search_query",
                                            "path", "filePath", "file_path", "file", "directory"))
    if has_content:
        return "write"
    if has_target and not has_content:
        return "read"
    return "unknown"


# ── Main Classifier ──────────────────────────────────────────────────────────

def classify_step(
    step: StepClassification,
    ctx: ClassifierContext,
) -> BaseType:
    """Classify a single step into an XEPV base type.

    Priority chain:
      0. Pre-assigned base (if adapter already determined it)
      1. V checks — verification patterns (highest heuristic priority)
      2. X checks — known read/explore tools + shell explore commands
      3. Unknown tool fallback — parameter shape + tool-name pattern
      4. E default

    After classification, call is recorded into ``ctx`` automatically.

    Args:
        step: The step to classify.
        ctx: Execution-level context (mutated in-place).

    Returns:
        The classified BaseType.
    """
    # Pre-assigned by adapter (e.g., P from LLM metadata)
    if step.pre_assigned_base is not None:
        ctx._record(step, step.pre_assigned_base)
        return step.pre_assigned_base

    resource = _extract_resource(step)
    if resource:
        step.resource = resource  # cache for later use

    tool = step.tool_name
    cmd = step.shell_command or ""
    is_known = tool in READ_TOOLS or tool in WRITE_TOOLS or tool in EXPLORE_TOOLS or tool == "runCmd"

    # ── V (Verify) ──────────────────────────────────────────────────────
    # Write-then-read same resource
    if tool in READ_TOOLS and resource:
        if any(r == resource for r, _ in ctx.recent_writes):
            return _finalize(step, BaseType.V, ctx)

    # Retry after failure
    if ctx.last_step and ctx.last_step.tool_name == tool and ctx.last_step.status == "error":
        return _finalize(step, BaseType.V, ctx)

    # Compile / test / lint after a write
    if cmd:
        if any(p.search(cmd) for p in VERIFY_CMD_PATTERNS) and ctx.recent_writes:
            return _finalize(step, BaseType.V, ctx)

    # ── X (Explore) ─────────────────────────────────────────────────────
    # Read on never-accessed resource
    if tool in READ_TOOLS and resource and resource not in ctx.successful_resources:
        return _finalize(step, BaseType.X, ctx)

    # Web search / fetch
    if tool in EXPLORE_TOOLS:
        return _finalize(step, BaseType.X, ctx)

    # Exploratory shell commands
    if cmd:
        if any(p.search(cmd.strip()) for p in EXPLORE_CMD_PATTERNS):
            return _finalize(step, BaseType.X, ctx)

    # ── Unknown tool fallback ───────────────────────────────────────────
    if not is_known:
        intent = _infer_intent_from_args(step.args)
        if intent == "read":
            return _finalize(step, BaseType.X, ctx)
        if intent == "unknown" and any(p.search(tool) for p in EXPLORE_NAME_PATTERNS):
            return _finalize(step, BaseType.X, ctx)

    # ── E (Execute) — default ───────────────────────────────────────────
    return _finalize(step, BaseType.E, ctx)


def _finalize(step: StepClassification, base: BaseType, ctx: ClassifierContext) -> BaseType:
    ctx._record(step, base)
    return base


# ── Convenience ──────────────────────────────────────────────────────────────

def classify_sequence(
    steps: list[StepClassification],
    ctx: ClassifierContext | None = None,
) -> list[BaseType]:
    """Classify a full sequence of steps, returning ordered base types."""
    if ctx is None:
        ctx = create_context()
    return [classify_step(s, ctx) for s in steps]
