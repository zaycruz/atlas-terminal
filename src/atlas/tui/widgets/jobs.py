"""Job status widget for displaying active agent tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static, ProgressBar
from textual.containers import Vertical


@dataclass
class JobInfo:
    """Information about a running job."""
    id: str
    name: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float | None = None  # 0.0 - 1.0
    message: str = ""


class JobStatusWidget(Static):
    """Widget displaying active jobs and their progress."""

    DEFAULT_CSS = """
    JobStatusWidget {
        background: $surface;
        border: solid $primary;
        padding: 1;
        height: auto;
        max-height: 12;
    }

    JobStatusWidget .widget-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    JobStatusWidget .job-item {
        margin-bottom: 1;
    }
    """

    jobs: reactive[Dict[str, JobInfo]] = reactive({}, always_update=True)

    def compose(self):
        yield Static("ACTIVE JOBS", classes="widget-title")
        yield Vertical(id="job-list")

    def watch_jobs(self, jobs: Dict[str, JobInfo]) -> None:
        """React to job changes."""
        container = self.query_one("#job-list", Vertical)
        container.remove_children()

        if not jobs:
            container.mount(Static(Text("No active jobs", style="dim")))
            return

        for job in jobs.values():
            icon = self._get_status_icon(job.status)
            style = self._get_status_style(job.status)

            job_text = Text()
            job_text.append(f"{icon} ", style=style)
            job_text.append(job.name)

            if job.message:
                job_text.append(f" - {job.message}", style="dim")

            container.mount(Static(job_text, classes="job-item"))

            if job.progress is not None and job.status == "running":
                bar = ProgressBar(total=100, show_percentage=True)
                bar.progress = int(job.progress * 100)
                container.mount(bar)

    def _get_status_icon(self, status: str) -> str:
        return {
            "queued": "○",
            "running": "◐",
            "completed": "●",
            "failed": "✗",
        }.get(status, "?")

    def _get_status_style(self, status: str) -> str:
        return {
            "queued": "dim",
            "running": "yellow",
            "completed": "green",
            "failed": "red",
        }.get(status, "")

    def add_job(self, job: JobInfo) -> None:
        """Add or update a job."""
        current = dict(self.jobs)
        current[job.id] = job
        self.jobs = current

    def update_job(
        self,
        job_id: str,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """Update an existing job."""
        if job_id not in self.jobs:
            return

        current = dict(self.jobs)
        job = current[job_id]

        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message

        self.jobs = current

    def remove_job(self, job_id: str) -> None:
        """Remove a job from the display."""
        current = dict(self.jobs)
        current.pop(job_id, None)
        self.jobs = current

    def clear_completed(self) -> None:
        """Remove all completed and failed jobs."""
        current = {
            k: v for k, v in self.jobs.items()
            if v.status not in ("completed", "failed")
        }
        self.jobs = current
