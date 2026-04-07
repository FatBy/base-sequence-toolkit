# Base Sequence Toolkit

**XEPV Base Sequence Analysis Framework for AI Agent Execution Traces**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## What is this?

When an AI agent solves a task, it executes a sequence of actions: reading files, writing code, running tests, searching for information. **Base Sequence Toolkit** classifies each action into one of four *base types*:

| Base | Name | Description | Examples |
|------|------|-------------|----------|
| **X** | eXplore | Information gathering | `readFile`, `webSearch`, `ls`, `grep` |
| **E** | Execute | State-changing actions | `writeFile`, `edit`, `npm install` |
| **P** | Plan | Pure reasoning/strategy | LLM thinking without tool calls |
| **V** | Verify | Validation & testing | `pytest`, `tsc --noEmit`, read-after-write |

A task execution becomes a **base sequence** like `XEEVXEV` — a compact fingerprint of the agent's behavioral strategy.

This toolkit provides:
- **Classifier**: Map tool calls → XEPV base types (stateful, context-aware)
- **Analyzer**: N-gram patterns, transition matrices, positional effects, risk profiles
- **Governor**: Three-layer adaptive intervention system (rule engine + statistics + self-adaptation)
- **Adapters**: Ready-made integrations for SWE-agent and DunCrew trace formats
- **CLI**: One-command analysis from terminal

## Key Findings

Analysis of 2,000+ SWE-agent trajectories and 500+ DunCrew execution traces revealed:

| Finding | Detail |
|---------|--------|
| **V-base deficit** | V (Verify) comprises only ~3.3% of all bases in SWE-agent traces. Resolved tasks have significantly more V bases than unresolved ones. |
| **E→V transition bottleneck** | The probability of transitioning from E (Execute) to V (Verify) is only 0.6% — agents almost never verify after executing. |
| **Exploration spirals** | Consecutive X runs (XXX+) are strongly associated with task failure. |
| **Late planning is harmful** | P bases appearing in the second half of execution correlate with lower success rates. |
| **E-V pairing predicts success** | The "golden path" pattern (Execute then Verify) is the strongest positive predictor. |

## Installation

```bash
pip install base-sequence-toolkit

# With SWE-agent HuggingFace support:
pip install base-sequence-toolkit[swe-agent]

# For development:
pip install base-sequence-toolkit[dev]
```

Or install from source:

```bash
git clone https://github.com/FatBy/base-sequence-toolkit.git
cd base-sequence-toolkit
pip install -e ".[dev,swe-agent]"
```

## Quick Start

### Analyze SWE-agent Trajectories

```python
from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface
from base_sequence_toolkit.core.analyzer import run_full_analysis

# Load and classify 500 SWE-agent trajectories
results = load_from_huggingface(max_records=500)

# Extract sequences and outcomes
sequences = [r.base_sequence for r in results]
outcomes = [r.resolved for r in results]

# Run comprehensive analysis
report = run_full_analysis(sequences, outcomes)
print(report.format_summary())
```

### Classify Your Own Agent's Actions

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification, BaseType

ctx = create_context()

steps = [
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),
    StepClassification(tool_name="writeFile", args={"path": "src/main.py", "content": "..."}, status="success"),
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),  # read-after-write → V
    StepClassification(tool_name="runCmd", shell_command="pytest tests/"),
]

for step in steps:
    base = classify_step(step, ctx)
    print(f"{step.tool_name:15s} → {base}")

# Output:
# readFile        → X  (first access to unknown file)
# writeFile       → E  (state-changing action)
# readFile        → V  (reading file we just wrote)
# runCmd          → V  (running tests after writes)
```

### Analyze Custom Sequences

```python
from base_sequence_toolkit.core.analyzer import (
    compute_sequence_stats,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
)

# Compute stats for a single sequence
stats = compute_sequence_stats("XEEVXEV")
print(f"Length: {stats.length}")
print(f"EV pairs: {stats.ev_pairs}")
print(f"Has verification: {stats.has_verification}")
print(f"Switch rate: {stats.switch_rate}")

# Transition matrix across multiple sequences
matrix = compute_transition_matrix(["XEEVXEV", "XXXEEE", "XEVEV"])
print(matrix.format())

# Risk profile
risk = extract_risk_profile("XXXEEEEE")
print(f"Risk flags: {risk.flags()}")
# → ['consecutive_x≥3', 'no_verification']

# Find patterns that distinguish success from failure
patterns = find_discriminative_patterns(
    resolved_sequences=["XEVEV", "XEEVE"],
    unresolved_sequences=["XXXEE", "XXXXX"],
    min_count=1,
)
for p in patterns[:5]:
    print(f"  {p.pattern}: lift={p.lift:.2f}")
```

## Governor: Adaptive Intervention System

The Governor is a three-layer closed-loop regulator that monitors agent execution in real-time and injects corrective prompts when behavioral anti-patterns are detected. **Zero LLM dependency** — pure rule-based logic.

### Architecture

| Layer | Name | Function | Latency |
|-------|------|----------|---------|
| **Layer 1** | Online Rule Engine | 7 rules + 8-dim feature vector → prompt injection | 0ms |
| **Layer 2** | Statistical Accumulator | 72-bucket success tracking + A/B intervention records | O(1) |
| **Layer 3** | Threshold Self-Adaptation | Chi-squared test → auto-adjust rule thresholds | Periodic |

### 7 Built-in Rules

| Rule | Description |
|------|-------------|
| `consecutive_x_brake` | Stops exploration spirals (XXX...) |
| `step_length_fuse` | Step budget guard (disabled in v4 — data showed >15 steps = 97.4% success) |
| `switch_rate_warning` | Detects erratic strategy switching |
| `diversity_collapse` | Catches strategy entropy collapse (all same type) |
| `late_planning_warning` | Penalizes re-planning past halfway |
| `missing_verification` | Enforces P→V golden path (96.9% success rate) |
| `explore_dominance` | Flags excessive X/(X+E) ratio |

### 8-Dimensional Feature Vector

```
consecutive_x      — trailing consecutive X count
step_count         — total steps so far
x_ratio_last5      — X ratio in last 5 steps
switch_rate        — adjacent-different-base ratio
p_in_late_half     — P appears in second half
last_p_followed_by_v — most recent P followed by V
max_e_run_length   — longest consecutive E run
xe_ratio           — X / (X + E)
```

### Quick Start: Real-time Monitoring

```python
from base_sequence_toolkit import BaseSequenceGovernor, InterventionRecord

governor = BaseSequenceGovernor()

# Layer 1: Evaluate during execution (after each tool call)
bases = ["X", "E", "E", "X", "X", "X", "X"]
signal = governor.evaluate(bases)

if signal.triggered:
    print(f"Rules fired: {signal.triggered_rules}")
    print(f"Inject into LLM: {signal.prompt_injection}")
    # → inject signal.prompt_injection into LLM system prompt

# Access 8-dim features
print(f"Consecutive X: {signal.features.consecutive_x}")
print(f"XE ratio: {signal.features.xe_ratio}")
```

### Quick Start: Post-Execution Learning

```python
# Layer 2+3: After task completes, record outcome
adjustments = governor.record_trace(
    base_sequence="X-E-E-X-X-X-X",
    success=False,
    interventions=[
        InterventionRecord(
            rule="consecutive_x_brake",
            step_index=5,
            features=signal.features,
        )
    ],
)

# Layer 3 auto-adapts thresholds when enough data accumulates
if adjustments:
    print(f"Threshold adjustments: {adjustments}")

# View statistics
summary = governor.get_stats_summary()
print(f"Total traces: {summary['total_traces']}")
print(f"Buckets: {summary['bucket_count']}")
```

### Standalone Functions

```python
from base_sequence_toolkit import extract_features, evaluate_sequence, RuleThresholds

# Extract features without full Governor
features = extract_features(["X", "E", "V", "E", "X"])
print(f"Switch rate: {features.switch_rate}")
print(f"XE ratio: {features.xe_ratio}")

# Evaluate with custom thresholds
custom = RuleThresholds(consecutive_x_brake=3, switch_rate_warning=0.5)
signal = evaluate_sequence(["X", "X", "X", "X"], thresholds=custom)
print(f"Triggered: {signal.triggered_rules}")
```

## CLI Usage

```bash
# Analyze SWE-agent trajectories
bst-analyze swe-agent -n 500 -o results/

# Analyze DunCrew execution traces
bst-analyze duncrew /path/to/exec_traces/ -o results/

# Analyze pre-computed JSON data
bst-analyze json data/my_sequences.json
```

## Writing Your Own Adapter

To integrate with a new agent framework, implement a function that converts your trace format into `StepClassification` objects:

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification

def classify_my_agent_trace(trace: dict) -> str:
    """Convert your agent's trace into an XEPV base sequence."""
    ctx = create_context()
    bases = []

    for action in trace["actions"]:
        step = StepClassification(
            tool_name=action["tool"],
            args=action.get("arguments", {}),
            status="success" if action["succeeded"] else "error",
            shell_command=action.get("command"),
        )
        base = classify_step(step, ctx)
        bases.append(str(base))

    return "".join(bases)
```

## Project Structure

```
base-sequence-toolkit/
├── base_sequence_toolkit/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   ├── core/
│   │   ├── classifier.py         # Generic XEPV classifier
│   │   ├── analyzer.py           # Analysis primitives
│   │   └── governor.py           # Three-layer adaptive Governor
│   └── adapters/
│       ├── swe_agent.py          # SWE-agent trajectory adapter
│       └── duncrew.py            # DunCrew trace adapter
├── examples/
│   └── swe_agent_analysis.py     # End-to-end example
├── tests/
│   ├── test_classifier.py
│   ├── test_analyzer.py
│   ├── test_governor.py
│   └── test_swe_agent.py
├── data/                         # Pre-computed results (optional)
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{base_sequence_toolkit,
  title = {Base Sequence Toolkit: XEPV Analysis Framework for AI Agent Execution Traces},
  year = {2025},
  url = {https://github.com/FatBy/base-sequence-toolkit}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Related

- **XEPV Framework**: The base classification system described in our paper
- **DunCrew**: AI operating system with XEPV-based execution governance
- **SWE-bench**: Software engineering benchmark for AI agents
