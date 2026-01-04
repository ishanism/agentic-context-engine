"""Tests for ace.integrations.copilot module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import subprocess

from ace.integrations.copilot import (
    ACECopilot,
    CopilotResult,
    COPILOT_AVAILABLE,
    _check_copilot_available,
)
from ace import Skillbook


class TestCopilotAvailability:
    """Test Copilot CLI availability detection."""

    def test_check_copilot_available_returns_bool(self):
        """Should return a boolean value."""
        result = _check_copilot_available()
        assert isinstance(result, bool)

    def test_copilot_available_is_bool(self):
        """COPILOT_AVAILABLE should be a boolean."""
        assert isinstance(COPILOT_AVAILABLE, bool)


class TestCopilotResult:
    """Test CopilotResult dataclass."""

    def test_copilot_result_success(self):
        """Should create successful result."""
        result = CopilotResult(
            success=True,
            output="Command executed",
            execution_trace="Step 1: Run test",
            returncode=0,
        )

        assert result.success is True
        assert result.output == "Command executed"
        assert result.execution_trace == "Step 1: Run test"
        assert result.returncode == 0
        assert result.error is None

    def test_copilot_result_failure(self):
        """Should create failed result with error."""
        result = CopilotResult(
            success=False,
            output="",
            execution_trace="",
            returncode=1,
            error="Command failed",
        )

        assert result.success is False
        assert result.error == "Command failed"
        assert result.returncode == 1

    def test_copilot_result_timeout(self):
        """Should create timeout result."""
        result = CopilotResult(
            success=False,
            output="",
            execution_trace="",
            returncode=-1,
            error="Execution timed out after 600s",
        )

        assert result.success is False
        assert result.returncode == -1
        assert "timed out" in result.error.lower()


@patch("ace.integrations.copilot.COPILOT_AVAILABLE", True)
class TestACECopilotInit:
    """Test ACECopilot initialization."""

    def test_init_creates_working_dir(self, tmp_path):
        """Should create working directory if it doesn't exist."""
        work_dir = tmp_path / "test_workspace"

        agent = ACECopilot(working_dir=str(work_dir))

        assert work_dir.exists()
        assert work_dir.is_dir()

    def test_init_with_existing_skillbook(self, tmp_path):
        """Should initialize with existing skillbook."""
        work_dir = tmp_path / "workspace"
        skillbook = Skillbook()
        skillbook.add_skill("test", "Test strategy")

        agent = ACECopilot(working_dir=str(work_dir), skillbook=skillbook)

        assert len(list(agent.skillbook.skills())) == 1

    def test_init_with_skillbook_path(self, tmp_path):
        """Should load skillbook from file."""
        work_dir = tmp_path / "workspace"
        skillbook_path = tmp_path / "test_skillbook.json"

        # Create and save skillbook
        skillbook = Skillbook()
        skillbook.add_skill("test", "Test strategy")
        skillbook.save_to_file(str(skillbook_path))

        agent = ACECopilot(
            working_dir=str(work_dir), skillbook_path=str(skillbook_path)
        )

        assert len(list(agent.skillbook.skills())) == 1

    def test_init_creates_empty_skillbook_by_default(self, tmp_path):
        """Should create empty skillbook if none provided."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir))

        assert len(list(agent.skillbook.skills())) == 0

    def test_init_sets_learning_enabled_by_default(self, tmp_path):
        """Should enable learning by default."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir))

        assert agent.is_learning is True

    def test_init_can_disable_learning(self, tmp_path):
        """Should allow disabling learning."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir), is_learning=False)

        assert agent.is_learning is False

    def test_init_with_async_learning(self, tmp_path):
        """Should initialize async learning if requested."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir), async_learning=True)

        assert agent.async_learning is True
        assert agent._learning_thread is not None


@patch("ace.integrations.copilot.COPILOT_AVAILABLE", True)
class TestACECopilotExecution:
    """Test ACECopilot execution methods."""

    @patch("ace.integrations.copilot.subprocess.run")
    def test_execute_copilot_success(self, mock_run, tmp_path):
        """Should execute copilot successfully."""
        work_dir = tmp_path / "workspace"
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "$ ls -la\nList files in directory"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        agent = ACECopilot(working_dir=str(work_dir), is_learning=False)
        result = agent._execute_copilot("List files")

        assert result.success is True
        assert result.returncode == 0
        assert mock_run.called

    @patch("ace.integrations.copilot.subprocess.run")
    def test_execute_copilot_timeout(self, mock_run, tmp_path):
        """Should handle timeout gracefully."""
        work_dir = tmp_path / "workspace"
        mock_run.side_effect = subprocess.TimeoutExpired("gh", 600)

        agent = ACECopilot(working_dir=str(work_dir), is_learning=False)
        result = agent._execute_copilot("Long running task")

        assert result.success is False
        assert result.returncode == -1
        assert "timed out" in result.error.lower()

    @patch("ace.integrations.copilot.subprocess.run")
    def test_execute_copilot_exception(self, mock_run, tmp_path):
        """Should handle exceptions gracefully."""
        work_dir = tmp_path / "workspace"
        mock_run.side_effect = Exception("Command failed")

        agent = ACECopilot(working_dir=str(work_dir), is_learning=False)
        result = agent._execute_copilot("Test task")

        assert result.success is False
        assert result.returncode == -1
        assert "Command failed" in result.error


@patch("ace.integrations.copilot.COPILOT_AVAILABLE", True)
class TestACECopilotParsing:
    """Test output parsing methods."""

    def test_parse_copilot_output_with_commands(self, tmp_path):
        """Should parse command suggestions."""
        work_dir = tmp_path / "workspace"
        agent = ACECopilot(working_dir=str(work_dir))

        stdout = """$ git status
Check repository status

$ git add .
Stage all changes
"""

        trace, summary = agent._parse_copilot_output(stdout, "")

        assert "[Step 1]" in trace
        assert "[Step 2]" in trace
        assert "git status" in trace or "git add" in trace

    def test_parse_copilot_output_with_code_blocks(self, tmp_path):
        """Should parse code blocks."""
        work_dir = tmp_path / "workspace"
        agent = ACECopilot(working_dir=str(work_dir))

        stdout = """Here's how to do it:

```python
def hello():
    print("Hello")
```
"""

        trace, summary = agent._parse_copilot_output(stdout, "")

        assert "[Step 1]" in trace
        assert "Code block" in trace

    def test_parse_copilot_output_empty(self, tmp_path):
        """Should handle empty output."""
        work_dir = tmp_path / "workspace"
        agent = ACECopilot(working_dir=str(work_dir))

        trace, summary = agent._parse_copilot_output("", "")

        assert trace == "(No trace captured)"


@patch("ace.integrations.copilot.COPILOT_AVAILABLE", True)
class TestACECopilotUtilities:
    """Test utility methods."""

    def test_save_and_load_skillbook(self, tmp_path):
        """Should save and load skillbook."""
        work_dir = tmp_path / "workspace"
        skillbook_path = tmp_path / "skillbook.json"

        agent = ACECopilot(working_dir=str(work_dir))
        agent.skillbook.add_skill("test", "Test skill")
        agent.save_skillbook(str(skillbook_path))

        # Create new agent and load
        agent2 = ACECopilot(working_dir=str(work_dir))
        agent2.load_skillbook(str(skillbook_path))

        assert len(list(agent2.skillbook.skills())) == 1

    def test_enable_disable_learning(self, tmp_path):
        """Should enable and disable learning."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir), is_learning=False)
        assert agent.is_learning is False

        agent.enable_learning()
        assert agent.is_learning is True

        agent.disable_learning()
        assert agent.is_learning is False

    def test_get_strategies_empty(self, tmp_path):
        """Should return empty string for empty skillbook."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir))
        strategies = agent.get_strategies()

        assert strategies == ""

    def test_get_strategies_with_skills(self, tmp_path):
        """Should return formatted strategies."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir))
        agent.skillbook.add_skill("test", "Test strategy")
        strategies = agent.get_strategies()

        assert "Test strategy" in strategies


@patch("ace.integrations.copilot.COPILOT_AVAILABLE", True)
class TestACECopilotAsyncLearning:
    """Test async learning features."""

    def test_learning_stats_initial(self, tmp_path):
        """Should return initial stats."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir), async_learning=True)
        stats = agent.learning_stats

        assert stats["async_learning"] is True
        assert stats["tasks_submitted"] == 0
        assert stats["tasks_completed"] == 0
        assert stats["pending"] == 0

    def test_wait_for_learning_sync_mode(self, tmp_path):
        """Should return immediately in sync mode."""
        work_dir = tmp_path / "workspace"

        agent = ACECopilot(working_dir=str(work_dir), async_learning=False)
        result = agent.wait_for_learning(timeout=1.0)

        assert result is True


class TestCopilotNotAvailable:
    """Test behavior when Copilot is not available."""

    @patch("ace.integrations.copilot.COPILOT_AVAILABLE", False)
    def test_init_raises_when_not_available(self, tmp_path):
        """Should raise RuntimeError when Copilot not available."""
        work_dir = tmp_path / "workspace"

        with pytest.raises(RuntimeError, match="GitHub Copilot CLI not found"):
            ACECopilot(working_dir=str(work_dir))
