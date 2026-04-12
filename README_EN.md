# LLM-Smart-Water

**A "Smart Meter" for Your AI Agent -- Decode Agent Behavioral DNA with the XEPV Alphabet**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Your%20Agent%20Has%20a%20Genome-orange.svg)](#citation)

**[English](README_EN.md) | [中文](README.md)**

---

## TL;DR

> Your AI Agent jumps between exploring, executing, planning, and verifying all day long -- but have you ever wondered: **What is it actually doing? How efficient is it? Is it wasting effort?**
>
> LLM-Smart-Water is a "smart meter" for the agent world -- it encodes every agent action into one of 4 letters (X, E, P, V), lines them up like DNA bases into a sequence, and automatically reveals your agent's behavioral habits, efficiency bottlenecks, and failure risks.

---

## What Problem Does This Solve?

AI Agents are everywhere now -- ReAct, Reflexion, SWE-agent, autonomous coding assistants -- they read files, write code, run tests, search the web. But here's the awkward truth:

**We only look at outcomes, not the process.**

Two agents both complete 90% of tasks -- equally good? Not necessarily. One might efficiently "glance, write, test" and be done. The other might "search endlessly, think and rethink, edit and re-edit" before stumbling to completion. Same outcome, but the second one will likely fail when facing novel problems.

**LLM-Smart-Water is the tool that lets you see the "process" clearly.**

### Core Idea: Turn Agent Behavior into a "Genome"

Inspired by biology: DNA encodes the entire blueprint of life with 4 base letters (A, T, C, G). We use a similar idea, encoding every AI agent action with 4 letters:

| Letter | Meaning | Plain English | Typical Actions |
|--------|---------|---------------|-----------------|
| **X** | eXplore | Agent is "looking around", "gathering info" | readFile, webSearch, ls, grep |
| **E** | Execute | Agent is "doing work", "changing things" | writeFile, edit, npm install, API calls |
| **P** | Plan | Agent is "thinking", "strategizing" | Pure reasoning, task decomposition, Reflexion |
| **V** | Verify | Agent is "checking results", "running tests" | pytest, tsc --noEmit, read-after-write |

A task execution becomes a **behavioral sequence** -- e.g., `XEEVXEV` (looked around -> wrote code -> wrote more -> verified -> looked again -> wrote -> verified).

**This is your agent's "behavioral genome", and LLM-Smart-Water is the tool that reads it.**

---

## What Did We Discover? (Key Research Findings)

We analyzed **2,347+ real agent execution traces** (347 from the production DunCrew system, 2,000 from open-source SWE-agent) and found striking patterns:

### Finding 1: Agents Almost Never Check Their Own Work

> **E -> V transition probability is only 2.1%**

After executing an action, agents only have a 2% chance of verifying the result. Like a developer who never runs tests after writing code -- this is a systemic problem in our data. In SWE-agent data, resolved traces have an E->V transition rate **nearly 2x** that of unresolved ones (54.2% vs. 28.1%).

**Bottom line: Agents that test after coding succeed far more than those that don't.**

### Finding 2: "Overthinking" is the Strongest Failure Signal

> **P ratio (planning proportion) correlates with success at r = -0.256, p < 0.0001**

Planning isn't bad per se -- but **too much planning relative to execution is the clearest marker of failure**. In failed tasks, planning ratio is **2x** that of successful ones (23.4% vs. 11.9%), while execution ratio is only **half** (23.4% vs. 38.4%).

**Bottom line: When an agent gets trapped in a "plan-explore-replan" loop, it's basically doomed.**

### Finding 3: P-X-P is the Only "High-Risk Pattern"

> **When P-X-P appears, success rate drops 10.4% (83.3% vs. 92.5% global)**

P-X-P means the agent "planned -> explored -> planned again", indicating the first planning attempt failed to incorporate exploration results, forcing a re-plan. Among all trigram patterns tested, this is the **only** one with statistically significant negative impact.

Interestingly, consecutive exploration X-X-X has a success rate (94.4%) **above** average -- the problem isn't too much exploration, but **failing to move from exploration to execution, instead looping back to planning**.

### Finding 4: Different Models Have Different "Behavioral Fingerprints"

Cross-model analysis on SWE-bench revealed striking differences:

| Model | Explore (X) | Execute (E) | Verify (V) | Resolve Rate |
|-------|------------|------------|------------|--------------|
| Llama-405B | 34.0% | 39.9% | **26.1%** | **42.5%** |
| Llama-70B | 43.0% | 40.0% | 17.0% | 16.2% |
| Llama-8B | 44.8% | 38.2% | 17.0% | 18.0% |

**Larger models naturally verify more (V ratio 50%+ higher) and resolve far more tasks.** Behavioral sequences serve as a model's "behavioral ID card".

---

## Governor: A Zero-Cost Agent "Coach"

Discovering problems isn't enough -- you need to fix them. So we built **Governor** -- a real-time system that monitors agent behavior and "nudges" it at critical moments.

### How Does It Work?

```
Agent executes a step -> Governor examines the sequence -> Detects a risky pattern -> Injects a corrective prompt -> Agent adjusts behavior
```

**Zero LLM overhead** -- Governor is pure rule-based logic, no extra model calls, latency < 1ms.

### Three-Layer Architecture

| Layer | What It Does | Analogy |
|-------|-------------|---------|
| **Layer 1: Rule Engine** | 7 rules + 8-dim feature vector; injects prompts when problems detected | "Real-time coach" -- spots issues immediately |
| **Layer 2: Statistical Accumulator** | Records intervention outcomes, tracks stats by feature buckets | "Data recorder" -- continuously collects feedback |
| **Layer 3: Self-Adaptation** | Periodically runs chi-squared tests to evaluate rule effectiveness, auto-adjusts thresholds | "Strategy optimizer" -- refines coaching methods based on data |

### 7 Built-in Rules (All Data-Driven, Not Hand-Waved)

| Rule | What It Does | Evidence |
|------|-------------|----------|
| `x_brake` | Stops prolonged exploration spirals | X->X self-loop at 57.2%, spirals endlessly without intervention |
| `miss_verify` | Reminds to verify after multiple executions | E->V only 2.1%, agents inherently skip verification |
| `switch_warn` | Warns against erratic strategy switching | switch_rate negatively correlates with success (r=-0.134) |
| `explore_dom` | Pushes toward execution when exploring too much | X/(X+E) imbalance signals agent is "spinning wheels" |
| `late_plan` | Urges execution when replanning past halfway | Late planning negatively correlates with success |
| `div_collapse` | Breaks loops when behavior becomes monotonic | Diversity collapse signals a dead loop |
| `step_fuse` | ~~Terminates overly long tasks~~ (disabled) | Data showed >15 steps = 97.4% success -- length isn't the problem! |

> The `step_fuse` story is fascinating: we initially assumed long tasks = failure, but the data told us the exact opposite. Governor's Layer 3 detected this rule was hurting performance, flagged it automatically, and it was disabled after human review. This is the value of data-driven rule management.

### Real-World Results

In production (101 pre-Governor vs. 246 post-Governor traces):

| Metric | Pre-Governor | Post-Governor | Change |
|--------|-------------|--------------|--------|
| **Task Success Rate** | 88.1% | **94.3%** | **+6.2%** |
| **Avg Token Cost** | 275K | **154K** | **-44%** |
| **Avg Steps** | 22.2 | **14.0** | **-37%** |

**Simultaneously improved success rate and reduced cost** -- not a trade-off, because Governor prevents wasteful exploration spirals (the primary source of token costs).

The `x_brake` rule alone contributed most of the gains: triggered in 60% of tasks, with a 98.6% success rate when fired.

---

## Installation

```bash
pip install base-sequence-toolkit

# With SWE-agent HuggingFace support:
pip install base-sequence-toolkit[swe-agent]

# For development:
pip install base-sequence-toolkit[dev]
```

From source:

```bash
git clone https://github.com/FatBy/base-sequence-toolkit.git
cd base-sequence-toolkit
pip install -e ".[dev,swe-agent]"
```

## Quick Start

### 1. Analyze SWE-agent Trajectories

```python
from base_sequence_toolkit.adapters.swe_agent import load_from_huggingface
from base_sequence_toolkit.core.analyzer import run_full_analysis

# Load 500 SWE-agent trajectories from HuggingFace and auto-classify
results = load_from_huggingface(max_records=500)

# Extract sequences and outcomes
sequences = [r.base_sequence for r in results]
outcomes = [r.resolved for r in results]

# Run full analysis: N-grams, transition matrices, risk profiles, etc.
report = run_full_analysis(sequences, outcomes)
print(report.format_summary())
```

### 2. Classify Your Own Agent's Actions

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification

ctx = create_context()

steps = [
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),
    StepClassification(tool_name="writeFile", args={"path": "src/main.py", "content": "..."}, status="success"),
    StepClassification(tool_name="readFile", args={"path": "src/main.py"}),  # read-after-write -> auto-classified as V
    StepClassification(tool_name="runCmd", shell_command="pytest tests/"),
]

for step in steps:
    base = classify_step(step, ctx)
    print(f"{step.tool_name:15s} -> {base}")

# Output:
# readFile        -> X  (first access, exploration)
# writeFile       -> E  (writing file, execution)
# readFile        -> V  (re-reading what we just wrote, verification!)
# runCmd          -> V  (running tests after writes, verification!)
```

### 3. Analyze Custom Sequences

```python
from base_sequence_toolkit.core.analyzer import (
    compute_sequence_stats,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
)

# Single sequence stats
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
# -> ['consecutive_x>=3', 'no_verification']

# Find patterns distinguishing success from failure
patterns = find_discriminative_patterns(
    resolved_sequences=["XEVEV", "XEEVE"],
    unresolved_sequences=["XXXEE", "XXXXX"],
    min_count=1,
)
for p in patterns[:5]:
    print(f"  {p.pattern}: lift={p.lift:.2f}")
```

### 4. Real-time Monitoring with Governor

```python
from base_sequence_toolkit import BaseSequenceGovernor, InterventionRecord

governor = BaseSequenceGovernor()

# Real-time evaluation (after each tool call)
bases = ["X", "E", "E", "X", "X", "X", "X"]
signal = governor.evaluate(bases)

if signal.triggered:
    print(f"Rules fired: {signal.triggered_rules}")
    print(f"Inject prompt: {signal.prompt_injection}")
    # -> inject signal.prompt_injection into the LLM system prompt

# Access 8-dim features
print(f"Consecutive X: {signal.features.consecutive_x}")
print(f"XE ratio: {signal.features.xe_ratio}")
```

```python
# After task completion, record outcome for Governor to learn
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

# Layer 3 auto-adjusts thresholds when enough data accumulates
if adjustments:
    print(f"Threshold adjustments: {adjustments}")
```

## CLI Usage

```bash
# Analyze SWE-agent trajectories
bst-analyze swe-agent -n 500 -o results/

# Analyze DunCrew execution traces
bst-analyze duncrew /path/to/exec_traces/ -o results/

# Analyze custom JSON data
bst-analyze json data/my_sequences.json
```

## Integrating Your Own Agent Framework

Implement an adapter to convert your agent's traces into XEPV sequences:

```python
from base_sequence_toolkit import classify_step, create_context, StepClassification

def classify_my_agent_trace(trace: dict) -> str:
    """Convert your agent's trace into an XEPV behavioral sequence."""
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

Adapters must satisfy two conditions:
1. **Completeness**: Every action in your agent maps to exactly one of X/E/P/V
2. **Semantic consistency**: Info gathering -> X, state changes -> E, validation -> V, pure reasoning -> P

## 8-Dimensional Feature Vector

Each behavioral sequence is extracted into 8 numerical features for real-time risk assessment:

```
consecutive_x        -- trailing consecutive X count (exploration inertia)
step_count           -- total steps (task complexity)
x_ratio_last5        -- X ratio in last 5 steps (local exploration density)
switch_rate          -- adjacent-different-type ratio (behavioral stability)
p_in_late_half       -- P appears in second half (late planning signal)
last_p_followed_by_v -- most recent P followed by V (verification coverage)
max_e_run_length     -- longest consecutive E run (execution momentum)
xe_ratio             -- X / (X + E) (exploration-execution ratio)
```

---

## Research Background: Deeper Insights

LLM-Smart-Water's theoretical foundation comes from the paper *"Your Agent Has a Genome: Sequence-Level Behavioral Analysis and Runtime Governance of LLM Autonomous Agents"*. Here are deeper insights from the paper:

### The "Cerebellum Hypothesis"

The paper proposes an architectural analogy:
- **LLM** is the agent's "cerebral cortex" -- responsible for reasoning, creativity, semantic understanding
- **Tool framework** (ReAct loop, skill system) provides "limbs" -- the ability to act on the world
- **Behavioral sequence governance (Governor)** is the "cerebellum" -- responsible for motor coordination and temporal sequencing

The human cerebellum contains ~69 billion neurons (more than the cerebral cortex) because motor coordination requires learning from vast amounts of movement experience. Similarly, a mature behavioral governance system needs to learn from massive execution traces to discover the complex patterns that determine success.

### Six Future Research Directions

| Direction | Description | Data Required |
|-----------|-------------|---------------|
| Behavioral Sequence LM | Model XEPV sequences as a "language" to predict next actions | ~3K-5K traces |
| Base-Conditioned Decoding | Use sequence signals to guide tool selection during LLM inference | ~50K traces |
| Sequence Anomaly Detection | Automatically discover unknown anomalous patterns | ~3K-5K traces |
| Dual-Stream Agent Architecture | Dedicated "sequence coordinator" running parallel to the LLM | ~1M+ traces |
| Behavioral Sequence Reward Model | Provide dense step-level rewards for agent RL training | ~50K traces |
| Behavioral Fingerprinting | Identify underlying models/frameworks from anonymous traces | Cross-system data |

### Cross-System Validation

We validated on 2,000 public SWE-agent traces from SWE-bench, confirming two core findings replicate across a completely different agent framework:
- **E->V verification deficit**: Resolved traces have 2x the E->V transition rate of unresolved ones
- **Exploration spirals**: Unresolved traces have 2.3x the average max consecutive exploration length (11.0 vs. 4.8)

This means these aren't bugs in a specific system -- they are **universal behavioral traits of LLM agents** -- autoregressive generation naturally tends to continue the current action type rather than proactively switching to verification.

---

## Project Structure

```
base-sequence-toolkit/
├── base_sequence_toolkit/
│   ├── __init__.py                 # Package entry, exports core API
│   ├── cli.py                      # CLI tool
│   ├── core/
│   │   ├── classifier.py           # XEPV classifier (stateful, context-aware)
│   │   ├── analyzer.py             # Analysis primitives: N-grams, transition matrices, risk profiles
│   │   └── governor.py             # Three-layer adaptive Governor
│   └── adapters/
│       ├── swe_agent.py            # SWE-agent trajectory adapter
│       └── duncrew.py              # DunCrew trace adapter
├── examples/
│   └── swe_agent_analysis.py       # End-to-end example
├── tests/                          # Test suite
├── data/                           # Pre-computed results (optional)
├── pyproject.toml
├── LICENSE
└── README.md
```

## Citation

If this tool is useful for your research, please cite:

```bibtex
@software{llm_smart_water,
  title = {LLM-Smart-Water: XEPV Behavioral Sequence Analysis Framework for AI Agent Execution Traces},
  author = {Deng, Sidi},
  year = {2026},
  url = {https://github.com/FatBy/base-sequence-toolkit}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Related

- **XEPV Framework** -- The behavioral sequence classification system described in the paper
- **DunCrew** -- AI Agent OS with integrated XEPV governance ([duncrew.com](https://duncrew.com))
- **SWE-bench** -- Software engineering benchmark for AI agents
