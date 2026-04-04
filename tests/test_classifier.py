"""Tests for core classifier module."""

from base_sequence_toolkit.core.classifier import (
    BaseType,
    ClassifierContext,
    StepClassification,
    classify_sequence,
    classify_step,
    create_context,
)


class TestBaseType:
    def test_string_representation(self):
        assert str(BaseType.X) == "X"
        assert str(BaseType.E) == "E"
        assert str(BaseType.P) == "P"
        assert str(BaseType.V) == "V"

    def test_enum_values(self):
        assert BaseType.X.value == "X"
        assert BaseType.E.value == "E"


class TestClassifyStep:
    def test_write_tool_classified_as_e(self):
        ctx = create_context()
        step = StepClassification(tool_name="writeFile", args={"path": "test.py", "content": "x"})
        assert classify_step(step, ctx) == BaseType.E

    def test_read_unknown_file_classified_as_x(self):
        ctx = create_context()
        step = StepClassification(tool_name="readFile", args={"path": "unknown.py"})
        assert classify_step(step, ctx) == BaseType.X

    def test_read_known_file_classified_as_e(self):
        ctx = create_context()
        # First access → X
        s1 = StepClassification(tool_name="readFile", args={"path": "known.py"}, status="success")
        classify_step(s1, ctx)
        # Second access → E (already known)
        s2 = StepClassification(tool_name="readFile", args={"path": "known.py"}, status="success")
        assert classify_step(s2, ctx) == BaseType.E

    def test_write_then_read_classified_as_v(self):
        ctx = create_context()
        # Write
        s1 = StepClassification(tool_name="writeFile", args={"path": "test.py", "content": "x"}, status="success")
        classify_step(s1, ctx)
        # Read same file → V
        s2 = StepClassification(tool_name="readFile", args={"path": "test.py"})
        assert classify_step(s2, ctx) == BaseType.V

    def test_retry_after_error_classified_as_v(self):
        ctx = create_context()
        s1 = StepClassification(tool_name="writeFile", args={"path": "x.py"}, status="error")
        classify_step(s1, ctx)
        s2 = StepClassification(tool_name="writeFile", args={"path": "y.py"})
        assert classify_step(s2, ctx) == BaseType.V

    def test_test_command_after_write_classified_as_v(self):
        ctx = create_context()
        s1 = StepClassification(tool_name="writeFile", args={"path": "x.py", "content": "x"}, status="success")
        classify_step(s1, ctx)
        s2 = StepClassification(tool_name="runCmd", args={}, shell_command="npm test")
        assert classify_step(s2, ctx) == BaseType.V

    def test_web_search_classified_as_x(self):
        ctx = create_context()
        step = StepClassification(tool_name="webSearch", args={"query": "python async"})
        assert classify_step(step, ctx) == BaseType.X

    def test_explore_cmd_classified_as_x(self):
        ctx = create_context()
        step = StepClassification(tool_name="runCmd", args={}, shell_command="ls -la")
        assert classify_step(step, ctx) == BaseType.X

    def test_pre_assigned_p_passed_through(self):
        ctx = create_context()
        step = StepClassification(tool_name="readFile", args={"path": "x"}, pre_assigned_base=BaseType.P)
        assert classify_step(step, ctx) == BaseType.P

    def test_unknown_tool_with_read_args_classified_as_x(self):
        ctx = create_context()
        step = StepClassification(tool_name="custom_tool", args={"url": "http://example.com"})
        assert classify_step(step, ctx) == BaseType.X

    def test_unknown_tool_with_write_args_classified_as_e(self):
        ctx = create_context()
        step = StepClassification(tool_name="custom_tool", args={"content": "hello"})
        assert classify_step(step, ctx) == BaseType.E

    def test_unknown_tool_with_search_name_classified_as_x(self):
        ctx = create_context()
        step = StepClassification(tool_name="searchDatabase", args={})
        assert classify_step(step, ctx) == BaseType.X

    def test_default_to_e(self):
        ctx = create_context()
        step = StepClassification(tool_name="unknownAction", args={})
        assert classify_step(step, ctx) == BaseType.E


class TestClassifySequence:
    def test_basic_sequence(self):
        steps = [
            StepClassification(tool_name="readFile", args={"path": "a.py"}),
            StepClassification(tool_name="writeFile", args={"path": "a.py", "content": "x"}, status="success"),
            StepClassification(tool_name="readFile", args={"path": "a.py"}),
        ]
        bases = classify_sequence(steps)
        assert bases == [BaseType.X, BaseType.E, BaseType.V]

    def test_empty_sequence(self):
        assert classify_sequence([]) == []


class TestPathNormalization:
    def test_backslash_normalization(self):
        ctx = create_context()
        s1 = StepClassification(tool_name="writeFile", args={"path": "src\\test.py", "content": "x"}, status="success")
        classify_step(s1, ctx)
        s2 = StepClassification(tool_name="readFile", args={"path": "src/test.py"})
        # Should match despite different separators
        assert classify_step(s2, ctx) == BaseType.V
