"""
Base Sequence Governor — Adaptive Closed-Loop Regulator for Agent Execution

Three-layer architecture, pure code, zero LLM dependency:

Layer 1 (Online Rule Engine):
  Evaluates base sequence entries after each ReAct loop turn.
  When triggered, returns prompt injection text to guide LLM strategy.
  7 rules + 8-dimensional feature vector.

Layer 2 (Statistical Accumulator):
  After each ExecTrace is saved, updates bucket statistics (success/total)
  and records intervention events for Layer 3 A/B comparison.

Layer 3 (Threshold Self-Adaptation):
  Triggered every N traces. Uses chi-squared test to compare intervention
  vs. control groups. Only adjusts thresholds when statistically significant.
  Includes counterfactual prediction for decision support.

Analogy: thermostat — sensor (classifier) + rules (if/else) + actuator (prompt injection)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal


# ── Feature Snapshot (8-dimensional) ─────────────────────────────────────────

@dataclass
class FeatureSnapshot:
    """8-dimensional feature vector, O(n) computable from base sequence."""
    # Original 4 dimensions
    consecutive_x: int = 0  # trailing consecutive X count
    step_count: int = 0  # total steps
    x_ratio_last5: float = 0.0  # X ratio in last 5 steps
    switch_rate: float = 0.0  # adjacent-different-base ratio

    # v2 additions (based on empirical findings)
    p_in_late_half: bool = False  # P appears in second half (77% vs 100%)
    last_p_followed_by_v: bool = False  # most recent P followed by V (96.9%)
    max_e_run_length: int = 0  # longest consecutive E run
    xe_ratio: float = 0.0  # X / (X + E)


# ── Rule Thresholds ──────────────────────────────────────────────────────────

@dataclass
class RuleThresholds:
    """Configurable thresholds for all 7 rules. Layer 3 can adjust these."""
    consecutive_x_brake: int = 12
    step_length_fuse: int = 12
    switch_rate_warning: float = 0.6
    adaptation_interval: int = 50
    # v2 additions
    diversity_collapse_window: int = 5
    late_planning_ratio: float = 0.5
    missing_verification_steps: int = 3
    explore_dominance_ratio: float = 0.55
    explore_dominance_min_steps: int = 6


DEFAULT_THRESHOLDS = RuleThresholds()

# Chi-squared critical value (df=1, alpha=0.05)
CHI_SQUARE_CRITICAL_005 = 3.841
MIN_SAMPLE_FOR_ADAPTATION = 20
MAX_PATTERN_LIBRARY_SIZE = 200
GAMMA = 0.9  # position discount decay factor

ALL_RULE_NAMES = [
    "consecutive_x_brake",
    "step_length_fuse",
    "switch_rate_warning",
    "diversity_collapse",
    "late_planning_warning",
    "missing_verification",
    "explore_dominance",
]


# ── Governor Signal ──────────────────────────────────────────────────────────

@dataclass
class GovernorSignal:
    """Layer 1 evaluation result."""
    triggered: bool
    prompt_injection: str
    triggered_rules: list[str]
    estimated_success_rate: float = -1.0
    features: FeatureSnapshot = field(default_factory=FeatureSnapshot)


@dataclass
class InterventionRecord:
    """Intervention event record (embedded in ExecTrace)."""
    rule: str
    step_index: int
    features: FeatureSnapshot
    counterfactual_success_rate: float = -1.0


# ── Bucket & Pattern Structures ──────────────────────────────────────────────

@dataclass
class BucketStats:
    success_count: int = 0
    total_count: int = 0


@dataclass
class PatternEntry:
    bucket_key: str
    sequence_snapshot: str
    rule: str
    success: bool
    recovery_path: str | None = None


@dataclass
class GovernorStats:
    """Persistent statistical data for all three layers."""
    version: int = 2
    buckets: dict[str, BucketStats] = field(default_factory=dict)
    intervention_effects: dict[str, dict[str, BucketStats]] = field(default_factory=dict)
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    total_trace_count: int = 0
    last_adaptation_count: int = 0
    pattern_library: list[PatternEntry] = field(default_factory=list)
    counterfactual_accumulator: dict[str, dict[str, float]] = field(default_factory=dict)


# ── Layer 1: Feature Extraction ──────────────────────────────────────────────

def extract_features(bases: list[str]) -> FeatureSnapshot:
    """Extract 8-dimensional feature vector from a base sequence.

    All computations are O(n), where n = sequence length (typically < 25).

    Args:
        bases: List of single-char bases, e.g. ["X", "E", "P", "V"].
    """
    n = len(bases)
    if n == 0:
        return FeatureSnapshot()

    # 1. Trailing consecutive X count
    consecutive_x = 0
    for i in range(n - 1, -1, -1):
        if bases[i] == "X":
            consecutive_x += 1
        else:
            break

    # 2. Total step count
    step_count = n

    # 3. X ratio in last 5 steps
    last5 = bases[-5:]
    x_ratio_last5 = sum(1 for b in last5 if b == "X") / len(last5) if last5 else 0.0

    # 4. Switch rate
    switch_count = sum(1 for i in range(1, n) if bases[i] != bases[i - 1])
    switch_rate = switch_count / (n - 1) if n > 1 else 0.0

    # 5. P in late half
    half_index = n // 2
    p_in_late_half = any(b == "P" for b in bases[half_index:])

    # 6. Last P followed by V
    last_p_followed_by_v = False
    for i in range(n - 1, -1, -1):
        if bases[i] == "P":
            last_p_followed_by_v = (i + 1 < n and bases[i + 1] == "V")
            break

    # 7. Max consecutive E run
    max_e_run = 0
    current_e_run = 0
    for b in bases:
        if b == "E":
            current_e_run += 1
            max_e_run = max(max_e_run, current_e_run)
        else:
            current_e_run = 0

    # 8. X / (X + E) ratio
    x_count = bases.count("X")
    e_count = bases.count("E")
    xe_ratio = x_count / (x_count + e_count) if (x_count + e_count) > 0 else 0.0

    return FeatureSnapshot(
        consecutive_x=consecutive_x,
        step_count=step_count,
        x_ratio_last5=round(x_ratio_last5, 4),
        switch_rate=round(switch_rate, 4),
        p_in_late_half=p_in_late_half,
        last_p_followed_by_v=last_p_followed_by_v,
        max_e_run_length=max_e_run,
        xe_ratio=round(xe_ratio, 4),
    )


# ── Layer 1: Rule Engine ─────────────────────────────────────────────────────

def evaluate_sequence(
    bases: list[str],
    thresholds: RuleThresholds | None = None,
) -> GovernorSignal:
    """Layer 1: Evaluate current base sequence, return intervention signal.

    Called after each tool execution in the ReAct loop.
    Pure if/else logic, 0ms latency, no model calls.

    Args:
        bases: List of base chars, e.g. ["X", "E", "P", "V", "E"].
        thresholds: Rule thresholds (uses defaults if None).

    Returns:
        GovernorSignal with triggered rules and prompt injection text.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    features = extract_features(bases)
    triggered_rules: list[str] = []
    injections: list[str] = []
    n = len(bases)

    # Rule 1: Consecutive X brake
    if features.consecutive_x >= thresholds.consecutive_x_brake:
        triggered_rules.append("consecutive_x_brake")
        if features.consecutive_x >= thresholds.consecutive_x_brake + 2:
            injections.append(
                f"[Base Sequence WARNING] You have performed {features.consecutive_x} consecutive "
                f"exploratory operations with no progress. Stop immediately, re-analyze the problem, "
                f"and formulate a completely different approach."
            )
        else:
            injections.append(
                f"[Base Sequence NOTICE] You have performed {features.consecutive_x} consecutive "
                f"exploratory operations. Stop and reconsider your strategy — consider changing "
                f"direction rather than continuing blind exploration."
            )

    # Rule 2: Step length fuse — DISABLED in v4
    # Data shows >15 steps has 97.4% success rate; length correlates positively
    # with success. This rule's assumption was inverted. Kept for reference.

    # Rule 3: Switch rate warning (only meaningful at >= 5 steps)
    if features.step_count >= 5 and features.switch_rate > thresholds.switch_rate_warning:
        triggered_rules.append("switch_rate_warning")
        injections.append(
            f"[Strategy Consistency] Your operations are switching between directions too frequently "
            f"(switch rate {features.switch_rate * 100:.0f}%). Focus on one direction and push deeper "
            f"rather than jumping back and forth."
        )

    # Rule 4: Diversity collapse (inspired by RAGEN's strategy entropy collapse)
    window = thresholds.diversity_collapse_window
    if n >= window * 2:
        recent = bases[-window:]
        prior = bases[-window * 2 : -window]
        recent_diversity = len(set(recent))
        prior_diversity = len(set(prior))
        if prior_diversity >= 3 and recent_diversity <= 1:
            triggered_rules.append("diversity_collapse")
            injections.append(
                f"[Diversity Collapse] Your last {window} operations are all the same type ({recent[0]}), "
                f"while previous operations were diverse. Your strategy may be stuck in a loop — "
                f"switch methods immediately."
            )

    # Rule 5: Late planning warning
    if (features.step_count > thresholds.step_length_fuse * thresholds.late_planning_ratio
            and n > 0 and bases[-1] == "P"):
        triggered_rules.append("late_planning_warning")
        injections.append(
            f"[Late Planning] Task is at step {features.step_count} (past halfway) and you're still "
            f"re-planning. Late-stage planning has significantly lower success rates. Execute with "
            f"available information rather than re-strategizing."
        )

    # Rule 6: Missing verification (P→V golden path)
    if n >= 2:
        dist_since_last_p = -1
        last_p_has_v = False
        for i in range(n - 1, -1, -1):
            if bases[i] == "P":
                dist_since_last_p = n - 1 - i
                last_p_has_v = (i + 1 < n and bases[i + 1] == "V")
                break
        if dist_since_last_p >= 2 and not last_p_has_v:
            triggered_rules.append("missing_verification")
            injections.append(
                f"[Missing Verification] You planned {dist_since_last_p} steps ago but haven't "
                f"verified results since. Data shows P->V path has 96.9% success rate. "
                f"Verify your current work immediately."
            )

    # Rule 7: Explore dominance
    if (features.xe_ratio > thresholds.explore_dominance_ratio
            and features.step_count > thresholds.explore_dominance_min_steps):
        triggered_rules.append("explore_dominance")
        injections.append(
            f"[Explore Dominance] Exploration operations are {features.xe_ratio * 100:.0f}% "
            f"of all actions ({features.step_count} steps), far above healthy levels. "
            f"Reduce exploration and execute with available information."
        )

    prompt_injection = "\n".join(injections) if injections else ""

    return GovernorSignal(
        triggered=len(triggered_rules) > 0,
        prompt_injection=prompt_injection,
        triggered_rules=triggered_rules,
        estimated_success_rate=-1.0,
        features=features,
    )


# ── Layer 2: Statistical Accumulator ─────────────────────────────────────────

def _to_bucket_key(features: FeatureSnapshot) -> str:
    """Map features to a coarse bucket key (72 possible buckets)."""
    cx = min(features.consecutive_x, 3)
    step = "S" if features.step_count <= 4 else ("M" if features.step_count <= 11 else "L")
    xr = "lo" if features.x_ratio_last5 < 0.4 else ("mi" if features.x_ratio_last5 <= 0.8 else "hi")
    sr = "L" if features.switch_rate <= 0.6 else "H"
    return f"{cx}_{step}_{xr}_{sr}"


def create_empty_stats() -> GovernorStats:
    """Create empty governor statistics."""
    return GovernorStats()


def _parse_sequence(seq: str) -> list[str]:
    """Parse 'X-E-P-V' or 'XEPV' into list of single chars."""
    if "-" in seq:
        return [b.strip() for b in seq.split("-") if b.strip() in "XEPV"]
    return [b for b in seq if b in "XEPV"]


def update_stats(
    stats: GovernorStats,
    base_sequence: str,
    success: bool,
    interventions: list[InterventionRecord],
) -> bool:
    """Layer 2: Update statistics from a completed trace.

    Args:
        stats: Current statistics (mutated in-place).
        base_sequence: Base sequence string, e.g. "X-E-E-V-X" or "XEEVX".
        success: Whether the task succeeded.
        interventions: Intervention records from this execution.

    Returns:
        True if Layer 3 adaptation should be triggered.
    """
    if not base_sequence:
        return False

    bases = _parse_sequence(base_sequence)
    if not bases:
        return False

    features = extract_features(bases)
    bucket_key = _to_bucket_key(features)

    # Update bucket statistics
    if bucket_key not in stats.buckets:
        stats.buckets[bucket_key] = BucketStats()
    stats.buckets[bucket_key].total_count += 1
    if success:
        stats.buckets[bucket_key].success_count += 1

    # Update intervention effects (covers all 7 rules)
    triggered_rule_names = {i.rule for i in interventions}
    for rule_name in ALL_RULE_NAMES:
        if rule_name not in stats.intervention_effects:
            stats.intervention_effects[rule_name] = {
                "intervened": BucketStats(),
                "control": BucketStats(),
            }
        effect = stats.intervention_effects[rule_name]
        if rule_name in triggered_rule_names:
            bucket = effect["intervened"]
        else:
            bucket = effect["control"]
        bucket.total_count += 1
        if success:
            bucket.success_count += 1

    # Counterfactual accumulator
    for intervention in interventions:
        rule = intervention.rule
        if rule not in stats.counterfactual_accumulator:
            stats.counterfactual_accumulator[rule] = {
                "sum_predicted": 0.0, "sum_actual": 0.0, "count": 0.0,
            }
        acc = stats.counterfactual_accumulator[rule]
        if intervention.counterfactual_success_rate >= 0:
            acc["sum_predicted"] += intervention.counterfactual_success_rate
            acc["sum_actual"] += 1.0 if success else 0.0
            acc["count"] += 1.0

    # Pattern library update
    for intervention in interventions:
        start = max(0, intervention.step_index - 5)
        seq_snapshot = "-".join(bases[start:intervention.step_index])
        recovery = bases[intervention.step_index:]
        stats.pattern_library.append(PatternEntry(
            bucket_key=bucket_key,
            sequence_snapshot=seq_snapshot,
            rule=intervention.rule,
            success=success,
            recovery_path="-".join(recovery) if recovery else None,
        ))
    # FIFO eviction
    if len(stats.pattern_library) > MAX_PATTERN_LIBRARY_SIZE:
        stats.pattern_library = stats.pattern_library[-MAX_PATTERN_LIBRARY_SIZE:]

    stats.total_trace_count += 1
    return (stats.total_trace_count - stats.last_adaptation_count) >= stats.thresholds.adaptation_interval


# ── Success Rate Lookup ──────────────────────────────────────────────────────

def lookup_success_rate(stats: GovernorStats, bases: list[str]) -> float:
    """Look up estimated success rate from bucket statistics.

    Uses position-discount weighted fallback when exact bucket has insufficient data.

    Returns:
        Success rate [0, 1], or -1 if insufficient data.
    """
    features = extract_features(bases)
    bucket_key = _to_bucket_key(features)
    bucket = stats.buckets.get(bucket_key)

    if bucket and bucket.total_count >= 3:
        return bucket.success_count / bucket.total_count

    # Fallback: weighted by proximity to current bucket
    cx = min(features.consecutive_x, 3)
    step_bucket = "S" if features.step_count <= 4 else ("M" if features.step_count <= 11 else "L")

    weighted_success = 0.0
    weighted_total = 0.0
    for key, value in stats.buckets.items():
        parts = key.split("_")
        if len(parts) < 2:
            continue
        try:
            key_cx = int(parts[0])
        except ValueError:
            continue
        key_step = parts[1]
        cx_dist = abs(key_cx - cx)
        step_match = 1.0 if key_step == step_bucket else 0.5
        weight = (GAMMA ** cx_dist) * step_match
        weighted_success += value.success_count * weight
        weighted_total += value.total_count * weight

    if weighted_total >= 3:
        return weighted_success / weighted_total

    return -1.0


def _query_pattern_library(
    patterns: list[PatternEntry],
    rule: str,
    bucket_key: str,
) -> str | None:
    """Query pattern library for historical recovery paths."""
    if not patterns:
        return None
    candidates = [p for p in patterns if p.rule == rule and p.success and p.recovery_path]
    if not candidates:
        return None
    exact = next((p for p in candidates if p.bucket_key == bucket_key), None)
    if exact:
        return exact.recovery_path
    return candidates[-1].recovery_path


# ── Layer 3: Threshold Self-Adaptation ───────────────────────────────────────

def _chi_square(
    intervention_success: int,
    intervention_fail: int,
    control_success: int,
    control_fail: int,
) -> float:
    """Yates-corrected chi-squared test for 2x2 contingency table."""
    a, b, c, d = intervention_success, intervention_fail, control_success, control_fail
    n = a + b + c + d
    if n == 0:
        return 0.0
    numerator = n * (abs(a * d - b * c) - n / 2) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _adapt_single_rule(
    stats: GovernorStats,
    rule_name: str,
    adjust_fn: callable,
) -> str | None:
    """Adapt a single rule's threshold based on intervention effectiveness."""
    if rule_name not in stats.intervention_effects:
        return None
    effect = stats.intervention_effects[rule_name]
    intervened = effect["intervened"]
    control = effect["control"]

    if intervened.total_count < MIN_SAMPLE_FOR_ADAPTATION or control.total_count < MIN_SAMPLE_FOR_ADAPTATION:
        return None

    intervention_rate = intervened.success_count / intervened.total_count
    control_rate = control.success_count / control.total_count

    chi2 = _chi_square(
        intervened.success_count,
        intervened.total_count - intervened.success_count,
        control.success_count,
        control.total_count - control.success_count,
    )

    if chi2 < CHI_SQUARE_CRITICAL_005:
        return None

    # Counterfactual hint
    cf_hint = ""
    cf_acc = stats.counterfactual_accumulator.get(rule_name)
    if cf_acc and cf_acc.get("count", 0) >= 10:
        avg_predicted = cf_acc["sum_predicted"] / cf_acc["count"]
        avg_actual = cf_acc["sum_actual"] / cf_acc["count"]
        delta = avg_actual - avg_predicted
        cf_hint = f", cf_delta={delta:.2f}"

    if intervention_rate > control_rate:
        result = adjust_fn("tighten")
        return f"{result} (effective, chi2={chi2:.1f}{cf_hint})" if result else None
    elif intervention_rate < control_rate:
        result = adjust_fn("loosen")
        return f"{result} (ineffective, chi2={chi2:.1f}{cf_hint})" if result else None
    return None


def adapt_thresholds(stats: GovernorStats) -> list[str]:
    """Layer 3: Self-adaptive threshold adjustment.

    Covers all 7 rules + counterfactual prediction support.

    Returns:
        List of adjustment descriptions (empty = no adjustments).
    """
    adjustments: list[str] = []
    t = stats.thresholds

    # Rule 1: consecutive_x_brake
    def adj1(d: str) -> str | None:
        if d == "tighten" and t.consecutive_x_brake > 1:
            t.consecutive_x_brake = max(1, t.consecutive_x_brake - 1)
            return f"consecutive_x_brake: tightened to {t.consecutive_x_brake}"
        if d == "loosen" and t.consecutive_x_brake < 12:
            t.consecutive_x_brake = min(12, t.consecutive_x_brake + 1)
            return f"consecutive_x_brake: loosened to {t.consecutive_x_brake}"
        return None
    r = _adapt_single_rule(stats, "consecutive_x_brake", adj1)
    if r: adjustments.append(r)

    # Rule 2: step_length_fuse
    def adj2(d: str) -> str | None:
        if d == "tighten" and t.step_length_fuse > 8:
            t.step_length_fuse = max(8, t.step_length_fuse - 2)
            return f"step_length_fuse: tightened to {t.step_length_fuse}"
        if d == "loosen" and t.step_length_fuse < 20:
            t.step_length_fuse = min(20, t.step_length_fuse + 2)
            return f"step_length_fuse: loosened to {t.step_length_fuse}"
        return None
    r = _adapt_single_rule(stats, "step_length_fuse", adj2)
    if r: adjustments.append(r)

    # Rule 3: switch_rate_warning
    def adj3(d: str) -> str | None:
        if d == "tighten" and t.switch_rate_warning > 0.4:
            t.switch_rate_warning = max(0.4, round(t.switch_rate_warning - 0.1, 1))
            return f"switch_rate_warning: tightened to {t.switch_rate_warning:.1f}"
        if d == "loosen" and t.switch_rate_warning < 0.8:
            t.switch_rate_warning = min(0.8, round(t.switch_rate_warning + 0.1, 1))
            return f"switch_rate_warning: loosened to {t.switch_rate_warning:.1f}"
        return None
    r = _adapt_single_rule(stats, "switch_rate_warning", adj3)
    if r: adjustments.append(r)

    # Rule 4: diversity_collapse
    def adj4(d: str) -> str | None:
        if d == "tighten" and t.diversity_collapse_window > 3:
            t.diversity_collapse_window = max(3, t.diversity_collapse_window - 1)
            return f"diversity_collapse: tightened window to {t.diversity_collapse_window}"
        if d == "loosen" and t.diversity_collapse_window < 8:
            t.diversity_collapse_window = min(8, t.diversity_collapse_window + 1)
            return f"diversity_collapse: loosened window to {t.diversity_collapse_window}"
        return None
    r = _adapt_single_rule(stats, "diversity_collapse", adj4)
    if r: adjustments.append(r)

    # Rule 5: late_planning_warning
    def adj5(d: str) -> str | None:
        if d == "tighten" and t.late_planning_ratio > 0.3:
            t.late_planning_ratio = max(0.3, round(t.late_planning_ratio - 0.1, 1))
            return f"late_planning_warning: tightened ratio to {t.late_planning_ratio:.1f}"
        if d == "loosen" and t.late_planning_ratio < 0.8:
            t.late_planning_ratio = min(0.8, round(t.late_planning_ratio + 0.1, 1))
            return f"late_planning_warning: loosened ratio to {t.late_planning_ratio:.1f}"
        return None
    r = _adapt_single_rule(stats, "late_planning_warning", adj5)
    if r: adjustments.append(r)

    # Rule 6: missing_verification
    def adj6(d: str) -> str | None:
        if d == "tighten" and t.missing_verification_steps > 2:
            t.missing_verification_steps = max(2, t.missing_verification_steps - 1)
            return f"missing_verification: tightened steps to {t.missing_verification_steps}"
        if d == "loosen" and t.missing_verification_steps < 6:
            t.missing_verification_steps = min(6, t.missing_verification_steps + 1)
            return f"missing_verification: loosened steps to {t.missing_verification_steps}"
        return None
    r = _adapt_single_rule(stats, "missing_verification", adj6)
    if r: adjustments.append(r)

    # Rule 7: explore_dominance
    def adj7(d: str) -> str | None:
        if d == "tighten" and t.explore_dominance_ratio > 0.5:
            t.explore_dominance_ratio = max(0.5, round(t.explore_dominance_ratio - 0.1, 1))
            return f"explore_dominance: tightened ratio to {t.explore_dominance_ratio:.1f}"
        if d == "loosen" and t.explore_dominance_ratio < 0.9:
            t.explore_dominance_ratio = min(0.9, round(t.explore_dominance_ratio + 0.1, 1))
            return f"explore_dominance: loosened ratio to {t.explore_dominance_ratio:.1f}"
        return None
    r = _adapt_single_rule(stats, "explore_dominance", adj7)
    if r: adjustments.append(r)

    stats.last_adaptation_count = stats.total_trace_count
    return adjustments


# ── Governor Service ─────────────────────────────────────────────────────────

class BaseSequenceGovernor:
    """Complete governor service combining all three layers.

    Usage:
        governor = BaseSequenceGovernor()

        # Layer 1: evaluate during execution
        signal = governor.evaluate(["X", "E", "E", "X", "X"])
        if signal.triggered:
            # inject signal.prompt_injection into LLM context
            pass

        # Layer 2+3: record after execution completes
        should_adapt = governor.record_trace("X-E-E-X-X", success=False, interventions=[...])
    """

    def __init__(self, stats: GovernorStats | None = None):
        self.stats = stats or create_empty_stats()

    def evaluate(self, bases: list[str]) -> GovernorSignal:
        """Layer 1: Evaluate current sequence and return intervention signal."""
        signal = evaluate_sequence(bases, self.stats.thresholds)

        # Attach estimated success rate if sufficient data
        if self.stats.total_trace_count >= 10:
            signal.estimated_success_rate = lookup_success_rate(self.stats, bases)

        # Pattern library: attach historical recovery experience
        if signal.triggered and self.stats.pattern_library:
            bucket_key = _to_bucket_key(signal.features)
            hints: list[str] = []
            for rule in signal.triggered_rules:
                recovery = _query_pattern_library(self.stats.pattern_library, rule, bucket_key)
                if recovery:
                    hints.append(
                        f"[Historical] In similar {rule} situations, "
                        f"recovery path {recovery} led to success."
                    )
            if hints:
                signal.prompt_injection += "\n" + "\n".join(hints)

        return signal

    def record_trace(
        self,
        base_sequence: str,
        success: bool,
        interventions: list[InterventionRecord] | None = None,
    ) -> list[str]:
        """Layer 2+3: Record a completed trace and possibly adapt thresholds.

        Returns:
            List of threshold adjustments made (empty = no changes).
        """
        if interventions is None:
            interventions = []
        should_adapt = update_stats(self.stats, base_sequence, success, interventions)
        if should_adapt:
            return adapt_thresholds(self.stats)
        return []

    def get_stats_summary(self) -> dict:
        """Get statistics summary for display."""
        summary: dict = {
            "total_traces": self.stats.total_trace_count,
            "bucket_count": len(self.stats.buckets),
            "thresholds": {
                "consecutive_x_brake": self.stats.thresholds.consecutive_x_brake,
                "switch_rate_warning": self.stats.thresholds.switch_rate_warning,
                "explore_dominance_ratio": self.stats.thresholds.explore_dominance_ratio,
                "diversity_collapse_window": self.stats.thresholds.diversity_collapse_window,
            },
            "intervention_summary": {},
        }
        for rule, effect in self.stats.intervention_effects.items():
            iv = effect["intervened"]
            ct = effect["control"]
            i_rate = f"{iv.success_count / iv.total_count * 100:.1f}%" if iv.total_count > 0 else "N/A"
            c_rate = f"{ct.success_count / ct.total_count * 100:.1f}%" if ct.total_count > 0 else "N/A"
            summary["intervention_summary"][rule] = {
                "intervention_rate": i_rate,
                "control_rate": c_rate,
                "sample_size": iv.total_count + ct.total_count,
            }
        return summary
