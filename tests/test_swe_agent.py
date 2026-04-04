"""Tests for SWE-agent adapter."""

from base_sequence_toolkit.adapters.swe_agent import (
    classify_swe_step,
    classify_trajectory,
    extract_command,
)
from base_sequence_toolkit.core.classifier import BaseType


class TestExtractCommand:
    def test_single_code_block(self):
        text = "Let me search for the file.\n```\nfind_file test.py\n```"
        assert extract_command(text) == "find_file test.py"

    def test_multiple_code_blocks(self):
        text = "First:\n```\nls\n```\nThen:\n```\nedit file.py\n```"
        assert extract_command(text) == "edit file.py"

    def test_no_code_block(self):
        text = "I need to think about this approach."
        assert extract_command(text) is None

    def test_empty_input(self):
        assert extract_command("") is None
        assert extract_command(None) is None


class TestClassifySweStep:
    def test_search_is_explore(self):
        assert classify_swe_step("```\nfind_file test.py\n```") == BaseType.X

    def test_edit_is_execute(self):
        assert classify_swe_step("```\nedit src/main.py\n```") == BaseType.E

    def test_pytest_is_verify(self):
        assert classify_swe_step("```\npytest tests/\n```") == BaseType.V

    def test_submit_is_verify(self):
        assert classify_swe_step("```\nsubmit\n```") == BaseType.V

    def test_no_command_is_plan(self):
        assert classify_swe_step("I need to think about the approach.") == BaseType.P

    def test_ls_is_explore(self):
        assert classify_swe_step("```\nls -la\n```") == BaseType.X

    def test_python_test_is_verify(self):
        assert classify_swe_step("```\npython -m pytest tests/\n```") == BaseType.V

    def test_python_print_is_explore(self):
        assert classify_swe_step("```\npython -c \"print(x)\"\n```") == BaseType.X

    def test_python_script_is_execute(self):
        assert classify_swe_step("```\npython setup.py install\n```") == BaseType.E

    def test_cat_is_explore(self):
        assert classify_swe_step("```\ncat README.md\n```") == BaseType.X

    def test_git_is_execute(self):
        assert classify_swe_step("```\ngit commit -m 'fix'\n```") == BaseType.E

    def test_npm_test_is_verify(self):
        assert classify_swe_step("```\nnpm test\n```") == BaseType.V


class TestClassifyTrajectory:
    def test_basic_trajectory(self):
        record = {
            "instance_id": "test-001",
            "model_name": "gpt-4",
            "target": True,
            "trajectory": [
                {"role": "ai", "text": "```\nfind_file test.py\n```"},
                {"role": "environment", "text": "Found: test.py"},
                {"role": "ai", "text": "```\nedit test.py\n```"},
                {"role": "environment", "text": "Edit successful"},
                {"role": "ai", "text": "```\npytest tests/\n```"},
                {"role": "environment", "text": "All tests passed"},
            ],
        }
        result = classify_trajectory(record)
        assert result.base_sequence == "XEV"
        assert result.step_count == 3
        assert result.resolved is True
        assert result.base_counts["X"] == 1
        assert result.base_counts["E"] == 1
        assert result.base_counts["V"] == 1

    def test_environment_messages_ignored(self):
        record = {
            "instance_id": "test-002",
            "model_name": "claude",
            "target": False,
            "trajectory": [
                {"role": "ai", "text": "```\nls\n```"},
                {"role": "environment", "text": "file1 file2"},
            ],
        }
        result = classify_trajectory(record)
        assert result.step_count == 1
        assert result.base_sequence == "X"
