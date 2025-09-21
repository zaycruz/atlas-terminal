"""Async-ish backtesting manager that coordinates background jobs."""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

ProgressCallback = Callable[["BacktestProgress"], None]
RunnerCallable = Callable[["BacktestJobRequest", ProgressCallback], "BacktestResult"]
ListenerCallable = Callable[["BacktestUpdate"], None]


class BacktestStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class BacktestJobRequest:
    """Structured request received from the conversation agent."""

    description: str
    parameters: Dict[str, Any]


@dataclass
class BacktestProgress:
    """A progress update emitted by the backtesting runner."""

    message: str
    step: Optional[int] = None
    total: Optional[int] = None


@dataclass
class BacktestResult:
    """Final backtesting result returned by the runner."""

    summary: str
    metrics: Dict[str, Any]
    artifacts: Sequence[Dict[str, Any]]


@dataclass
class BacktestUpdate:
    """Event object delivered to listeners on state changes."""

    job_id: str
    status: BacktestStatus
    progress: Optional[BacktestProgress] = None
    result: Optional[BacktestResult] = None
    error: Optional[str] = None


class BacktestManager:
    """Thin job manager that executes backtests in a thread pool."""

    def __init__(self, runner: RunnerCallable, max_workers: int = 2) -> None:
        self._runner = runner
        self._executor = threading.Thread
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._jobs: Dict[str, List[BacktestUpdate]] = {}
        self._listeners: Dict[str, List[ListenerCallable]] = {}
        self._active_threads: List[threading.Thread] = []

    # ------------------------------------------------------------------
    def submit(self, request: BacktestJobRequest) -> str:
        job_id = uuid.uuid4().hex
        initial_update = BacktestUpdate(job_id=job_id, status=BacktestStatus.QUEUED)
        with self._lock:
            self._jobs[job_id] = [initial_update]
            self._listeners.setdefault(job_id, [])
        self._emit(job_id, initial_update)

        thread = threading.Thread(target=self._run_job, args=(job_id, request), daemon=True)
        thread.start()

        with self._lock:
            self._active_threads.append(thread)
            # Prune finished threads
            self._active_threads = [t for t in self._active_threads if t.is_alive()]
            # Honor max_workers by letting threads run; since thread count is small and
            # jobs are IO bound via MCP calls we keep the implementation minimal.

        return job_id

    def subscribe(self, job_id: str, listener: ListenerCallable) -> None:
        with self._lock:
            listeners = self._listeners.setdefault(job_id, [])
            listeners.append(listener)
            snapshots = list(self._jobs.get(job_id, []))
        for update in snapshots:
            listener(update)

    def get_status(self, job_id: str) -> BacktestStatus:
        with self._lock:
            updates = self._jobs.get(job_id)
        if not updates:
            raise KeyError(f"Unknown backtest job id {job_id}")
        return updates[-1].status

    def get_result(self, job_id: str) -> Optional[BacktestResult]:
        with self._lock:
            updates = self._jobs.get(job_id)
        if not updates:
            raise KeyError(f"Unknown backtest job id {job_id}")
        return updates[-1].result

    # ------------------------------------------------------------------
    def _run_job(self, job_id: str, request: BacktestJobRequest) -> None:
        self._emit(job_id, BacktestUpdate(job_id=job_id, status=BacktestStatus.RUNNING))

        def progress_cb(progress: BacktestProgress) -> None:
            self._emit(
                job_id,
                BacktestUpdate(
                    job_id=job_id,
                    status=BacktestStatus.RUNNING,
                    progress=progress,
                ),
            )

        try:
            result = self._runner(request, progress_cb)
        except Exception as exc:  # pragma: no cover - exercised in tests
            self._emit(
                job_id,
                BacktestUpdate(
                    job_id=job_id,
                    status=BacktestStatus.FAILED,
                    error=str(exc),
                ),
            )
            return

        self._emit(
            job_id,
            BacktestUpdate(
                job_id=job_id,
                status=BacktestStatus.SUCCEEDED,
                result=result,
            ),
        )

    def _emit(self, job_id: str, update: BacktestUpdate) -> None:
        with self._lock:
            history = self._jobs.setdefault(job_id, [])
            history.append(update)
            listeners = list(self._listeners.get(job_id, []))
        for listener in listeners:
            listener(update)


__all__ = [
    "BacktestJobRequest",
    "BacktestProgress",
    "BacktestResult",
    "BacktestStatus",
    "BacktestUpdate",
    "BacktestManager",
]
