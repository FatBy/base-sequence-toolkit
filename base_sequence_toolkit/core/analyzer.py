"""
XEPV Sequence Analyzer

Provides analysis primitives for base sequences:
  - Base distribution & ratios
  - N-gram frequency analysis
  - Transition probability matrix (4x4)
  - Positional effects (where each base appears)
  - Run-length statistics (consecutive same-base runs)
  - Discriminative pattern mining (resolved vs. unresolved)
  - Risk feature extraction
  - Temporal trend analysis
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from base_sequence_toolkit.core.classifier import BaseType

BASES = ("X", "E", "P", "V")


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BaseDistribution:
    """Count and ratio for each base type."""
    X: int = 0
    E: int = 0
    P: int = 0
    V: int = 0

    @property
    def total(self) -> int:
        return self.X + self.E + self.P + self.V

    def ratio(self, base: str) -> float:
        t = self.total
        return getattr(self, base) / t if t > 0 else 0.0

    def ratios(self) -> dict[str, float]:
        return {b: self.ratio(b) for b in BASES}

    def as_dict(self) -> dict[str, int]:
        return {b: getattr(self, b) for b in BASES}


@dataclass
class TransitionMatrix:
    """4x4 transition probability matrix between XEPV bases."""
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def probability(self, from_base: str, to_base: str) -> float:
        row_total = sum(self.counts.get(from_base + b, 0) for b in BASES)
        return self.counts.get(from_base + to_base, 0) / row_total if row_total > 0 else 0.0

    def as_matrix(self) -> dict[str, dict[str, float]]:
        return {fb: {tb: self.probability(fb, tb) for tb in BASES} for fb in BASES}

    def format(self) -> str:
        lines = [f"{'':>4s}" + "".join(f"  →{b:>5s}" for b in BASES)]
        for fb in BASES:
            row_total = sum(self.counts.get(fb + b, 0) for b in BASES)
            row = f"  {fb} "
            for tb in BASES:
                pct = self.probability(fb, tb) * 100
                row += f"  {pct:5.1f}%"
            row += f"  (n={row_total})"
            lines.append(row)
        return "\n".join(lines)


@dataclass
class SequenceStats:
    """Comprehensive statistics for a single base sequence."""
    sequence: str  # e.g. "XEPVEE"
    length: int
    distribution: BaseDistribution
    first_base: str
    last_base: str
    max_consecutive: dict[str, int]  # max run per base
    ev_pairs: int  # E immediately followed by V
    switch_rate: float  # fraction of adjacent pairs that differ
    x_segments: int  # number of contiguous X runs
    has_verification: bool  # whether V appears at all


@dataclass
class NgramResult:
    """N-gram pattern with resolved/unresolved statistics."""
    pattern: str
    resolved_count: int
    unresolved_count: int
    total: int
    resolved_rate: float
    unresolved_rate: float
    lift: float  # resolved_rate / unresolved_rate


@dataclass
class PositionalAnalysis:
    """Positional distribution of a specific base type."""
    base: str
    has_base_count: int  # traces that have this base
    no_base_count: int  # traces without
    avg_first_position: float | None  # normalized [0,1]
    avg_all_positions: float | None  # normalized [0,1]
    position_histogram: dict[int, int]  # decile bucket → count


@dataclass
class RiskProfile:
    """Risk features extracted from a sequence."""
    consecutive_x_gte_3: bool
    no_verification: bool
    length_over_10: bool
    x_ratio_over_50: bool
    switch_rate_over_06: bool
    late_planning: bool  # P in the second half

    def flags(self) -> list[str]:
        result = []
        if self.consecutive_x_gte_3:
            result.append("consecutive_x≥3")
        if self.no_verification:
            result.append("no_verification")
        if self.length_over_10:
            result.append("length>10")
        if self.x_ratio_over_50:
            result.append("x_ratio>50%")
        if self.switch_rate_over_06:
            result.append("switch_rate>0.6")
        if self.late_planning:
            result.append("late_planning")
        return result


# ── Builders ─────────────────────────────────────────────────────────────────

def build_sequence_string(
    bases: Sequence[BaseType | str],
    separator: str = "",
) -> str:
    """Build a sequence string from base types. Use separator="" for compact form."""
    return separator.join(str(b) for b in bases)


def build_distribution(bases: Sequence[BaseType | str]) -> BaseDistribution:
    """Count each base type in a sequence."""
    dist = BaseDistribution()
    for b in bases:
        s = str(b)
        if hasattr(dist, s):
            setattr(dist, s, getattr(dist, s) + 1)
    return dist


def parse_sequence(seq: str) -> list[str]:
    """Parse a sequence string into a list of single-char bases.

    Handles both "XEPV" (compact) and "X-E-P-V" (dash-separated) formats.
    """
    if "-" in seq:
        return [b.strip() for b in seq.split("-") if b.strip()]
    return list(seq)


# ── Sequence Statistics ──────────────────────────────────────────────────────

def compute_sequence_stats(seq: str) -> SequenceStats:
    """Compute comprehensive statistics for a base sequence string."""
    bases = parse_sequence(seq)
    n = len(bases)
    if n == 0:
        return SequenceStats(
            sequence=seq, length=0, distribution=BaseDistribution(),
            first_base="", last_base="", max_consecutive={},
            ev_pairs=0, switch_rate=0.0, x_segments=0, has_verification=False,
        )

    dist = build_distribution(bases)

    # Max consecutive runs per base
    max_cons: dict[str, int] = {b: 0 for b in BASES}
    current_run = 1
    for i in range(1, n):
        if bases[i] == bases[i - 1]:
            current_run += 1
        else:
            b = bases[i - 1]
            max_cons[b] = max(max_cons.get(b, 0), current_run)
            current_run = 1
    max_cons[bases[-1]] = max(max_cons.get(bases[-1], 0), current_run)

    # E-V pairs
    ev_pairs = sum(1 for i in range(n - 1) if bases[i] == "E" and bases[i + 1] == "V")

    # Switch rate
    switches = sum(1 for i in range(n - 1) if bases[i] != bases[i + 1])
    switch_rate = switches / (n - 1) if n > 1 else 0.0

    # X segments
    x_segments = 0
    for i in range(n):
        if bases[i] == "X" and (i == 0 or bases[i - 1] != "X"):
            x_segments += 1

    return SequenceStats(
        sequence=seq,
        length=n,
        distribution=dist,
        first_base=bases[0],
        last_base=bases[-1],
        max_consecutive=max_cons,
        ev_pairs=ev_pairs,
        switch_rate=round(switch_rate, 4),
        x_segments=x_segments,
        has_verification="V" in bases,
    )


# ── N-gram Analysis ─────────────────────────────────────────────────────────

def compute_ngrams(sequence: str, n: int) -> Counter[str]:
    """Compute n-gram frequencies from a compact base sequence string.

    Args:
        sequence: Compact sequence like "XEPVEE" (no separators).
        n: N-gram size (2, 3, or 4 recommended).

    Returns:
        Counter mapping each n-gram to its count.
    """
    bases = parse_sequence(sequence)
    flat = "".join(bases)
    ngrams: Counter[str] = Counter()
    for i in range(len(flat) - n + 1):
        ngrams[flat[i : i + n]] += 1
    return ngrams


# ── Transition Matrix ────────────────────────────────────────────────────────

def compute_transition_matrix(sequences: Sequence[str]) -> TransitionMatrix:
    """Compute aggregate transition matrix from multiple sequences.

    Args:
        sequences: List of compact base sequences (e.g. ["XEPVEE", "XXEE"]).
    """
    matrix = TransitionMatrix()
    for seq in sequences:
        bases = parse_sequence(seq)
        for i in range(len(bases) - 1):
            pair = bases[i] + bases[i + 1]
            matrix.counts[pair] += 1
    return matrix


# ── Discriminative Pattern Mining ────────────────────────────────────────────

def find_discriminative_patterns(
    resolved_sequences: Sequence[str],
    unresolved_sequences: Sequence[str],
    ngram_sizes: tuple[int, ...] = (2, 3, 4),
    min_count: int = 10,
) -> list[NgramResult]:
    """Find n-gram patterns that discriminate between resolved and unresolved traces.

    Args:
        resolved_sequences: Compact sequences from successful executions.
        unresolved_sequences: Compact sequences from failed executions.
        ngram_sizes: Which n-gram sizes to analyze.
        min_count: Minimum total occurrences to include a pattern.

    Returns:
        List of NgramResult sorted by lift (descending).
    """
    pattern_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"resolved": 0, "unresolved": 0})

    for seq in resolved_sequences:
        for n in ngram_sizes:
            for gram, count in compute_ngrams(seq, n).items():
                pattern_counts[gram]["resolved"] += count

    for seq in unresolved_sequences:
        for n in ngram_sizes:
            for gram, count in compute_ngrams(seq, n).items():
                pattern_counts[gram]["unresolved"] += count

    n_res = max(len(resolved_sequences), 1)
    n_unres = max(len(unresolved_sequences), 1)

    results: list[NgramResult] = []
    for pattern, counts in pattern_counts.items():
        total = counts["resolved"] + counts["unresolved"]
        if total < min_count:
            continue
        r_rate = counts["resolved"] / n_res
        u_rate = counts["unresolved"] / n_unres
        lift = r_rate / max(u_rate, 0.001)
        results.append(NgramResult(
            pattern=pattern,
            resolved_count=counts["resolved"],
            unresolved_count=counts["unresolved"],
            total=total,
            resolved_rate=round(r_rate, 4),
            unresolved_rate=round(u_rate, 4),
            lift=round(lift, 3),
        ))

    results.sort(key=lambda r: -r.lift)
    return results


# ── Positional Analysis ─────────────────────────────────────────────────────

def analyze_positions(
    sequences: Sequence[str],
    target_base: str = "V",
) -> PositionalAnalysis:
    """Analyze where a specific base type tends to appear in sequences.

    Positions are normalized to [0, 1] where 0 = start, 1 = end.

    Args:
        sequences: Compact base sequences.
        target_base: Which base to analyze (default "V").

    Returns:
        PositionalAnalysis with distribution data.
    """
    all_positions: list[float] = []
    first_positions: list[float] = []
    histogram: dict[int, int] = defaultdict(int)  # decile → count
    has_count = 0

    for seq in sequences:
        bases = parse_sequence(seq)
        n = len(bases)
        if n == 0:
            continue

        found_first = False
        for i, b in enumerate(bases):
            if b == target_base:
                norm_pos = i / max(n - 1, 1)
                all_positions.append(norm_pos)
                decile = min(int(norm_pos * 10), 9)
                histogram[decile] += 1
                if not found_first:
                    first_positions.append(norm_pos)
                    found_first = True

        if found_first:
            has_count += 1

    no_count = len(sequences) - has_count

    return PositionalAnalysis(
        base=target_base,
        has_base_count=has_count,
        no_base_count=no_count,
        avg_first_position=round(sum(first_positions) / len(first_positions), 4) if first_positions else None,
        avg_all_positions=round(sum(all_positions) / len(all_positions), 4) if all_positions else None,
        position_histogram=dict(histogram),
    )


# ── Risk Profile ─────────────────────────────────────────────────────────────

def extract_risk_profile(seq: str) -> RiskProfile:
    """Extract risk features from a base sequence.

    These features correspond to empirically validated risk signals
    from DunCrew execution data and SWE-agent trajectory analysis.
    """
    bases = parse_sequence(seq)
    n = len(bases)
    flat = "".join(bases)

    # Consecutive X ≥ 3
    consecutive_x = "XXX" in flat

    # No V verification
    no_v = "V" not in bases

    # Length > 10
    long = n > 10

    # X ratio > 50%
    x_count = bases.count("X")
    x_heavy = (x_count / n > 0.5) if n > 0 else False

    # Switch rate > 0.6
    switches = sum(1 for i in range(n - 1) if bases[i] != bases[i + 1])
    switch_rate = switches / (n - 1) if n > 1 else 0.0
    high_switch = switch_rate > 0.6

    # Late planning: P in second half
    half = n // 2
    late_p = any(b == "P" for b in bases[half:]) if n > 2 else False

    return RiskProfile(
        consecutive_x_gte_3=consecutive_x,
        no_verification=no_v,
        length_over_10=long,
        x_ratio_over_50=x_heavy,
        switch_rate_over_06=high_switch,
        late_planning=late_p,
    )


# ── Aggregate Analysis ───────────────────────────────────────────────────────

@dataclass
class AggregateReport:
    """Full analysis report for a collection of execution traces."""
    total_traces: int
    resolved_count: int
    unresolved_count: int
    overall_distribution: BaseDistribution
    resolved_distribution: BaseDistribution
    unresolved_distribution: BaseDistribution
    transition_matrix_resolved: TransitionMatrix
    transition_matrix_unresolved: TransitionMatrix
    v_positional: PositionalAnalysis
    discriminative_patterns: list[NgramResult]
    risk_flag_rates: dict[str, dict[str, float]]  # flag → {rate, resolved_rate, unresolved_rate}

    def format_summary(self) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("XEPV BASE SEQUENCE ANALYSIS REPORT")
        lines.append("=" * 70)

        lines.append(f"\nTotal traces: {self.total_traces}")
        lines.append(f"Resolved: {self.resolved_count} ({100 * self.resolved_count / max(self.total_traces, 1):.1f}%)")
        lines.append(f"Unresolved: {self.unresolved_count}")

        lines.append("\n--- Overall Base Distribution ---")
        for b in BASES:
            r = self.overall_distribution.ratio(b)
            lines.append(f"  {b}: {getattr(self.overall_distribution, b):6d} ({r * 100:.1f}%)")

        lines.append("\n--- Resolved vs Unresolved ---")
        for label, dist in [("Resolved", self.resolved_distribution), ("Unresolved", self.unresolved_distribution)]:
            lines.append(f"  {label}:")
            for b in BASES:
                lines.append(f"    {b}: {dist.ratio(b) * 100:.1f}%")

        lines.append("\n--- Transition Matrix (Resolved) ---")
        lines.append(self.transition_matrix_resolved.format())

        lines.append("\n--- Transition Matrix (Unresolved) ---")
        lines.append(self.transition_matrix_unresolved.format())

        lines.append("\n--- V-Base Positional Analysis ---")
        vp = self.v_positional
        lines.append(f"  Traces with V: {vp.has_base_count} ({100 * vp.has_base_count / max(vp.has_base_count + vp.no_base_count, 1):.1f}%)")
        if vp.avg_first_position is not None:
            lines.append(f"  Avg first V position: {vp.avg_first_position:.2f} (0=start, 1=end)")

        lines.append("\n--- Top Discriminative Patterns (by lift) ---")
        for p in self.discriminative_patterns[:10]:
            lines.append(f"  {p.pattern:6s}  lift={p.lift:.2f}  resolved={p.resolved_count:4d}  unresolved={p.unresolved_count:4d}")

        lines.append("\n--- Risk Flag Rates ---")
        for flag, rates in self.risk_flag_rates.items():
            lines.append(f"  {flag}: overall={rates['rate'] * 100:.1f}%, "
                         f"resolved={rates['resolved_rate'] * 100:.1f}%, "
                         f"unresolved={rates['unresolved_rate'] * 100:.1f}%")

        return "\n".join(lines)


def run_full_analysis(
    sequences: Sequence[str],
    outcomes: Sequence[bool],
    min_pattern_count: int = 10,
) -> AggregateReport:
    """Run a complete XEPV analysis on a collection of traces.

    Args:
        sequences: Compact base sequences (e.g. ["XEPVEE", "XXEEE"]).
        outcomes: Corresponding success/failure flags (True = resolved).
        min_pattern_count: Minimum n-gram count for pattern mining.

    Returns:
        AggregateReport with all analysis results.
    """
    assert len(sequences) == len(outcomes), "sequences and outcomes must have same length"

    resolved_seqs = [s for s, o in zip(sequences, outcomes) if o]
    unresolved_seqs = [s for s, o in zip(sequences, outcomes) if not o]

    # Distributions
    all_bases_flat = [b for seq in sequences for b in parse_sequence(seq)]
    res_bases_flat = [b for seq in resolved_seqs for b in parse_sequence(seq)]
    unres_bases_flat = [b for seq in unresolved_seqs for b in parse_sequence(seq)]

    # Transition matrices
    tm_res = compute_transition_matrix(resolved_seqs)
    tm_unres = compute_transition_matrix(unresolved_seqs)

    # V positional analysis
    v_pos = analyze_positions(sequences, "V")

    # Discriminative patterns
    patterns = find_discriminative_patterns(resolved_seqs, unresolved_seqs, min_count=min_pattern_count)

    # Risk flag rates
    all_risks = [extract_risk_profile(s) for s in sequences]
    res_risks = [extract_risk_profile(s) for s in resolved_seqs]
    unres_risks = [extract_risk_profile(s) for s in unresolved_seqs]

    flag_names = ["consecutive_x≥3", "no_verification", "length>10",
                  "x_ratio>50%", "switch_rate>0.6", "late_planning"]
    flag_attrs = ["consecutive_x_gte_3", "no_verification", "length_over_10",
                  "x_ratio_over_50", "switch_rate_over_06", "late_planning"]

    risk_rates: dict[str, dict[str, float]] = {}
    for name, attr in zip(flag_names, flag_attrs):
        all_rate = sum(1 for r in all_risks if getattr(r, attr)) / max(len(all_risks), 1)
        res_rate = sum(1 for r in res_risks if getattr(r, attr)) / max(len(res_risks), 1)
        unres_rate = sum(1 for r in unres_risks if getattr(r, attr)) / max(len(unres_risks), 1)
        risk_rates[name] = {"rate": round(all_rate, 4), "resolved_rate": round(res_rate, 4), "unresolved_rate": round(unres_rate, 4)}

    return AggregateReport(
        total_traces=len(sequences),
        resolved_count=len(resolved_seqs),
        unresolved_count=len(unresolved_seqs),
        overall_distribution=build_distribution(all_bases_flat),
        resolved_distribution=build_distribution(res_bases_flat),
        unresolved_distribution=build_distribution(unres_bases_flat),
        transition_matrix_resolved=tm_res,
        transition_matrix_unresolved=tm_unres,
        v_positional=v_pos,
        discriminative_patterns=patterns,
        risk_flag_rates=risk_rates,
    )
