"""Tests for Governor module — all three layers."""

from base_sequence_toolkit.core.governor import (
    BaseSequenceGovernor,
    BucketStats,
    FeatureSnapshot,
    GovernorStats,
    InterventionRecord,
    RuleThresholds,
    adapt_thresholds,
    create_empty_stats,
    evaluate_sequence,
    extract_features,
    lookup_success_rate,
    update_stats,
)


# ── Feature Extraction ───────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_empty(self):
        f = extract_features([])
        assert f.step_count == 0
        assert f.consecutive_x == 0

    def test_trailing_x(self):
        f = extract_features(["E", "X", "X", "X"])
        assert f.consecutive_x == 3
        assert f.step_count == 4

    def test_no_trailing_x(self):
        f = extract_features(["X", "X", "E"])
        assert f.consecutive_x == 0

    def test_x_ratio_last5(self):
        f = extract_features(["E", "E", "X", "X", "X", "X", "X"])
        assert f.x_ratio_last5 == 1.0  # last 5 are all X

    def test_switch_rate(self):
        f = extract_features(["X", "E", "X", "E"])
        assert f.switch_rate == 1.0

    def test_no_switches(self):
        f = extract_features(["E", "E", "E"])
        assert f.switch_rate == 0.0

    def test_p_in_late_half(self):
        f = extract_features(["X", "E", "E", "P", "E"])
        assert f.p_in_late_half is True

    def test_p_only_in_early_half(self):
        f = extract_features(["P", "E", "E", "E", "E", "E"])
        assert f.p_in_late_half is False

    def test_last_p_followed_by_v(self):
        f = extract_features(["X", "P", "V", "E"])
        assert f.last_p_followed_by_v is True

    def test_last_p_not_followed_by_v(self):
        f = extract_features(["X", "P", "E", "E"])
        assert f.last_p_followed_by_v is False

    def test_max_e_run(self):
        f = extract_features(["X", "E", "E", "E", "X", "E", "E"])
        assert f.max_e_run_length == 3

    def test_xe_ratio(self):
        f = extract_features(["X", "X", "E", "E"])
        assert f.xe_ratio == 0.5


# ── Layer 1: Rule Evaluation ─────────────────────────────────────────────────

class TestEvaluateSequence:
    def test_no_trigger_on_normal_sequence(self):
        signal = evaluate_sequence(["X", "E", "V"])
        assert signal.triggered is False
        assert signal.prompt_injection == ""

    def test_consecutive_x_brake(self):
        # Default threshold is 12
        bases = ["X"] * 12
        signal = evaluate_sequence(bases)
        assert signal.triggered is True
        assert "consecutive_x_brake" in signal.triggered_rules

    def test_consecutive_x_not_triggered_below_threshold(self):
        bases = ["X"] * 11
        signal = evaluate_sequence(bases)
        assert "consecutive_x_brake" not in signal.triggered_rules

    def test_switch_rate_warning(self):
        # Switch rate = 1.0 with 5 steps
        bases = ["X", "E", "X", "E", "X"]
        signal = evaluate_sequence(bases)
        assert "switch_rate_warning" in signal.triggered_rules

    def test_switch_rate_not_triggered_short_sequence(self):
        bases = ["X", "E", "X"]
        signal = evaluate_sequence(bases)
        assert "switch_rate_warning" not in signal.triggered_rules

    def test_diversity_collapse(self):
        # 10+ bases, prior diverse, recent all same
        bases = ["X", "E", "P", "V", "E", "E", "E", "E", "E", "E"]
        signal = evaluate_sequence(bases)
        assert "diversity_collapse" in signal.triggered_rules

    def test_late_planning_warning(self):
        # stepLengthFuse=12, latePlanningRatio=0.5, so threshold = 6
        bases = ["X", "E", "E", "E", "E", "E", "E", "P"]
        signal = evaluate_sequence(bases)
        assert "late_planning_warning" in signal.triggered_rules

    def test_missing_verification(self):
        bases = ["X", "P", "E", "E", "E"]
        signal = evaluate_sequence(bases)
        assert "missing_verification" in signal.triggered_rules

    def test_missing_verification_not_triggered_with_v(self):
        bases = ["X", "P", "V", "E"]
        signal = evaluate_sequence(bases)
        assert "missing_verification" not in signal.triggered_rules

    def test_explore_dominance(self):
        # 7 steps, xe_ratio > 0.55
        bases = ["X", "X", "X", "X", "E", "X", "X"]
        signal = evaluate_sequence(bases)
        assert "explore_dominance" in signal.triggered_rules

    def test_custom_thresholds(self):
        thresholds = RuleThresholds(consecutive_x_brake=3)
        bases = ["X", "X", "X"]
        signal = evaluate_sequence(bases, thresholds)
        assert "consecutive_x_brake" in signal.triggered_rules

    def test_multiple_rules_can_trigger(self):
        # Long exploration spiral triggers multiple rules
        bases = ["X"] * 15
        signal = evaluate_sequence(bases)
        assert len(signal.triggered_rules) >= 2  # at least consecutive_x + explore_dominance


# ── Layer 2: Statistics Update ───────────────────────────────────────────────

class TestUpdateStats:
    def test_basic_update(self):
        stats = create_empty_stats()
        update_stats(stats, "X-E-V", True, [])
        assert stats.total_trace_count == 1
        assert len(stats.buckets) == 1

    def test_bucket_counting(self):
        stats = create_empty_stats()
        update_stats(stats, "X-E-V", True, [])
        update_stats(stats, "X-E-V", False, [])
        # Same sequence → same bucket
        bucket = list(stats.buckets.values())[0]
        assert bucket.total_count == 2
        assert bucket.success_count == 1

    def test_intervention_tracking(self):
        stats = create_empty_stats()
        intervention = InterventionRecord(
            rule="consecutive_x_brake",
            step_index=5,
            features=FeatureSnapshot(consecutive_x=5, step_count=5),
        )
        update_stats(stats, "X-X-X-X-X", False, [intervention])
        effect = stats.intervention_effects["consecutive_x_brake"]
        assert effect["intervened"].total_count == 1
        assert effect["control"].total_count == 0

    def test_adaptation_trigger(self):
        stats = create_empty_stats()
        stats.thresholds.adaptation_interval = 5
        for i in range(4):
            should = update_stats(stats, "X-E-V", True, [])
            assert should is False
        should = update_stats(stats, "X-E-V", True, [])
        assert should is True

    def test_pattern_library_fifo(self):
        stats = create_empty_stats()
        for i in range(210):
            intervention = InterventionRecord(
                rule="test_rule", step_index=0,
                features=FeatureSnapshot(),
            )
            update_stats(stats, "X-E", True, [intervention])
        assert len(stats.pattern_library) <= 200

    def test_empty_sequence(self):
        stats = create_empty_stats()
        result = update_stats(stats, "", True, [])
        assert result is False
        assert stats.total_trace_count == 0

    def test_compact_format(self):
        stats = create_empty_stats()
        update_stats(stats, "XEVEE", True, [])
        assert stats.total_trace_count == 1


# ── Success Rate Lookup ──────────────────────────────────────────────────────

class TestLookupSuccessRate:
    def test_insufficient_data(self):
        stats = create_empty_stats()
        rate = lookup_success_rate(stats, ["X", "E"])
        assert rate == -1.0

    def test_direct_bucket_hit(self):
        stats = create_empty_stats()
        for _ in range(5):
            update_stats(stats, "X-E-V", True, [])
        for _ in range(5):
            update_stats(stats, "X-E-V", False, [])
        rate = lookup_success_rate(stats, ["X", "E", "V"])
        assert 0.4 <= rate <= 0.6  # approximately 50%


# ── Layer 3: Threshold Adaptation ────────────────────────────────────────────

class TestAdaptThresholds:
    def test_no_adaptation_insufficient_data(self):
        stats = create_empty_stats()
        adjustments = adapt_thresholds(stats)
        assert adjustments == []

    def test_adaptation_with_significant_data(self):
        stats = create_empty_stats()
        # Simulate: intervention group has higher success rate
        stats.intervention_effects["consecutive_x_brake"] = {
            "intervened": BucketStats(success_count=18, total_count=20),
            "control": BucketStats(success_count=5, total_count=20),
        }
        adjustments = adapt_thresholds(stats)
        # Should detect significant difference and tighten
        assert any("consecutive_x_brake" in a for a in adjustments)

    def test_no_adaptation_when_not_significant(self):
        stats = create_empty_stats()
        stats.intervention_effects["consecutive_x_brake"] = {
            "intervened": BucketStats(success_count=10, total_count=20),
            "control": BucketStats(success_count=10, total_count=20),
        }
        adjustments = adapt_thresholds(stats)
        # Equal rates → not significant → no adjustment
        assert not any("consecutive_x_brake" in a for a in adjustments)


# ── Governor Service (integration) ───────────────────────────────────────────

class TestBaseSequenceGovernor:
    def test_evaluate_clean_sequence(self):
        gov = BaseSequenceGovernor()
        signal = gov.evaluate(["X", "E", "V"])
        assert signal.triggered is False

    def test_evaluate_risky_sequence(self):
        gov = BaseSequenceGovernor()
        signal = gov.evaluate(["X"] * 15)
        assert signal.triggered is True
        assert len(signal.prompt_injection) > 0

    def test_record_trace(self):
        gov = BaseSequenceGovernor()
        adjustments = gov.record_trace("X-E-V-E-V", success=True)
        assert isinstance(adjustments, list)
        assert gov.stats.total_trace_count == 1

    def test_full_lifecycle(self):
        gov = BaseSequenceGovernor()
        # Simulate 10 traces
        for i in range(10):
            bases = ["X", "E", "V", "E"]
            signal = gov.evaluate(bases)
            gov.record_trace("X-E-V-E", success=(i % 3 != 0))
        summary = gov.get_stats_summary()
        assert summary["total_traces"] == 10
        assert summary["bucket_count"] >= 1

    def test_stats_summary(self):
        gov = BaseSequenceGovernor()
        gov.record_trace("X-E-V", success=True)
        summary = gov.get_stats_summary()
        assert "total_traces" in summary
        assert "thresholds" in summary
        assert "intervention_summary" in summary
