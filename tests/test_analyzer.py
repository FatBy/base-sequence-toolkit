"""Tests for core analyzer module."""

from base_sequence_toolkit.core.analyzer import (
    BaseDistribution,
    TransitionMatrix,
    analyze_positions,
    build_distribution,
    build_sequence_string,
    compute_ngrams,
    compute_sequence_stats,
    compute_transition_matrix,
    extract_risk_profile,
    find_discriminative_patterns,
    parse_sequence,
    run_full_analysis,
)
from base_sequence_toolkit.core.classifier import BaseType


class TestParseSequence:
    def test_compact_format(self):
        assert parse_sequence("XEPV") == ["X", "E", "P", "V"]

    def test_dash_separated(self):
        assert parse_sequence("X-E-P-V") == ["X", "E", "P", "V"]

    def test_empty(self):
        assert parse_sequence("") == []


class TestBuildSequenceString:
    def test_from_base_types(self):
        bases = [BaseType.X, BaseType.E, BaseType.P, BaseType.V]
        assert build_sequence_string(bases) == "XEPV"

    def test_with_separator(self):
        bases = [BaseType.X, BaseType.E]
        assert build_sequence_string(bases, separator="-") == "X-E"


class TestBaseDistribution:
    def test_build_distribution(self):
        dist = build_distribution(["X", "E", "E", "V", "X"])
        assert dist.X == 2
        assert dist.E == 2
        assert dist.V == 1
        assert dist.P == 0
        assert dist.total == 5

    def test_ratios(self):
        dist = build_distribution(["E", "E", "E", "E"])
        assert dist.ratio("E") == 1.0
        assert dist.ratio("X") == 0.0

    def test_empty_distribution(self):
        dist = build_distribution([])
        assert dist.total == 0
        assert dist.ratio("E") == 0.0


class TestComputeNgrams:
    def test_bigrams(self):
        ngrams = compute_ngrams("XEPV", 2)
        assert ngrams["XE"] == 1
        assert ngrams["EP"] == 1
        assert ngrams["PV"] == 1

    def test_trigrams(self):
        ngrams = compute_ngrams("XEPV", 3)
        assert ngrams["XEP"] == 1
        assert ngrams["EPV"] == 1

    def test_repeated_pattern(self):
        ngrams = compute_ngrams("XEXE", 2)
        assert ngrams["XE"] == 2
        assert ngrams["EX"] == 1


class TestTransitionMatrix:
    def test_single_sequence(self):
        matrix = compute_transition_matrix(["XEPV"])
        assert matrix.probability("X", "E") == 1.0
        assert matrix.probability("E", "P") == 1.0
        assert matrix.probability("P", "V") == 1.0

    def test_self_transitions(self):
        matrix = compute_transition_matrix(["XXXX"])
        assert matrix.probability("X", "X") == 1.0

    def test_format(self):
        matrix = compute_transition_matrix(["XEPV"])
        formatted = matrix.format()
        assert "X" in formatted
        assert "%" in formatted


class TestSequenceStats:
    def test_basic_stats(self):
        stats = compute_sequence_stats("XEPVEE")
        assert stats.length == 6
        assert stats.first_base == "X"
        assert stats.last_base == "E"
        assert stats.ev_pairs == 0  # no E directly followed by V here
        assert stats.has_verification is True

    def test_ev_pairs(self):
        stats = compute_sequence_stats("XEVEV")
        assert stats.ev_pairs == 2

    def test_switch_rate(self):
        # All same → 0.0
        stats = compute_sequence_stats("EEEE")
        assert stats.switch_rate == 0.0
        # All different → 1.0
        stats = compute_sequence_stats("XEPV")
        assert stats.switch_rate == 1.0

    def test_max_consecutive(self):
        stats = compute_sequence_stats("XXXEEVVVV")
        assert stats.max_consecutive["X"] == 3
        assert stats.max_consecutive["E"] == 2
        assert stats.max_consecutive["V"] == 4

    def test_x_segments(self):
        stats = compute_sequence_stats("XEXEXXE")
        assert stats.x_segments == 3  # X, X, XX

    def test_dash_separated_input(self):
        stats = compute_sequence_stats("X-E-P-V")
        assert stats.length == 4


class TestDiscriminativePatterns:
    def test_finds_patterns(self):
        resolved = ["XEVEV", "XEVEE", "XEVVE"]
        unresolved = ["XXXEE", "XXXXX", "XXXEX"]
        patterns = find_discriminative_patterns(resolved, unresolved, min_count=1)
        assert len(patterns) > 0
        # EV should have high lift (more common in resolved)
        ev_patterns = [p for p in patterns if p.pattern == "EV"]
        assert len(ev_patterns) == 1
        assert ev_patterns[0].lift > 1.0


class TestPositionalAnalysis:
    def test_v_positions(self):
        sequences = ["XEPV", "XXVX", "EEEE"]
        result = analyze_positions(sequences, "V")
        assert result.has_base_count == 2
        assert result.no_base_count == 1
        assert result.avg_first_position is not None


class TestRiskProfile:
    def test_consecutive_x(self):
        profile = extract_risk_profile("XXXEE")
        assert profile.consecutive_x_gte_3 is True

    def test_no_verification(self):
        profile = extract_risk_profile("XEEXE")
        assert profile.no_verification is True

    def test_has_verification(self):
        profile = extract_risk_profile("XEVEE")
        assert profile.no_verification is False

    def test_long_sequence(self):
        profile = extract_risk_profile("XEEXEEXEEXEE")
        assert profile.length_over_10 is True

    def test_late_planning(self):
        profile = extract_risk_profile("XEEEPE")
        assert profile.late_planning is True

    def test_flags(self):
        profile = extract_risk_profile("XXXXEEEE")
        flags = profile.flags()
        assert "consecutive_x>=3" in flags or "consecutive_x≥3" in flags


class TestRunFullAnalysis:
    def test_basic_analysis(self):
        sequences = ["XEVEE", "XXEEV", "XXXEE", "XEVVE", "EEEEE"]
        outcomes = [True, True, False, True, False]
        report = run_full_analysis(sequences, outcomes, min_pattern_count=1)
        assert report.total_traces == 5
        assert report.resolved_count == 3
        assert report.unresolved_count == 2
        assert report.overall_distribution.total > 0

    def test_format_summary(self):
        sequences = ["XEVEE", "XXXEE"]
        outcomes = [True, False]
        report = run_full_analysis(sequences, outcomes, min_pattern_count=1)
        summary = report.format_summary()
        assert "XEPV BASE SEQUENCE ANALYSIS REPORT" in summary
        assert "Resolved" in summary
