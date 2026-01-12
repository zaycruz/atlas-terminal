"""Tests for TUI widgets."""

import pytest
from datetime import datetime

from atlas.tui.widgets.chat import ChatMessage


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="System notification")
        assert msg.role == "system"


# Note: Full widget testing requires Textual's testing framework
# These tests verify the data structures and logic that can be tested in isolation


class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_job_info_creation(self):
        from atlas.tui.widgets.jobs import JobInfo

        job = JobInfo(
            id="job1",
            name="Backtest",
            status="running",
            progress=0.5,
            message="Processing...",
        )
        assert job.id == "job1"
        assert job.name == "Backtest"
        assert job.status == "running"
        assert job.progress == 0.5
        assert job.message == "Processing..."

    def test_job_info_defaults(self):
        from atlas.tui.widgets.jobs import JobInfo

        job = JobInfo(
            id="j1",
            name="Test",
            status="pending",
        )
        assert job.progress is None
        assert job.message == ""


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        from atlas.tui.widgets.monitor import Alert

        ts = datetime.now()
        alert = Alert(
            id="alert1",
            type="news",
            title="Breaking News",
            timestamp=ts,
            priority=5,
            url="https://example.com",
        )
        assert alert.id == "alert1"
        assert alert.type == "news"
        assert alert.title == "Breaking News"
        assert alert.timestamp == ts
        assert alert.priority == 5
        assert alert.url == "https://example.com"

    def test_alert_without_url(self):
        from atlas.tui.widgets.monitor import Alert

        alert = Alert(
            id="a1",
            type="economic",
            title="Fed Decision",
            timestamp=datetime.now(),
            priority=10,
        )
        assert alert.url is None
