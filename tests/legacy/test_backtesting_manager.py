import threading
import time
import unittest

from atlas.backtesting import (
    BacktestJobRequest,
    BacktestManager,
    BacktestProgress,
    BacktestResult,
    BacktestStatus,
)


def dummy_runner(request: BacktestJobRequest, progress_cb):
    progress_cb(BacktestProgress(message="boot", step=0, total=2))
    time.sleep(0.01)
    progress_cb(BacktestProgress(message=f"run {request.description}", step=1, total=2))
    return BacktestResult(
        summary=f"finished {request.description}",
        metrics={"alpha": 0.1},
        artifacts=[{"type": "log", "content": "done"}],
    )


def failing_runner(request: BacktestJobRequest, progress_cb):
    progress_cb(BacktestProgress(message="starting"))
    raise RuntimeError("boom")


class BacktestManagerTests(unittest.TestCase):
    def test_success_flow_emits_progress_and_result(self):
        manager = BacktestManager(dummy_runner)
        request = BacktestJobRequest(description="ema", parameters={"symbol": "AAPL"})
        updates = []
        done = threading.Event()

        job_id = manager.submit(request)

        def listener(update):
            updates.append(update)
            if update.status in (BacktestStatus.SUCCEEDED, BacktestStatus.FAILED):
                done.set()

        manager.subscribe(job_id, listener)
        self.assertTrue(done.wait(timeout=1.0))

        statuses = [u.status for u in updates]
        self.assertIn(BacktestStatus.QUEUED, statuses)
        self.assertIn(BacktestStatus.RUNNING, statuses)
        self.assertIn(BacktestStatus.SUCCEEDED, statuses)

        result = manager.get_result(job_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.summary, "finished ema")
        self.assertEqual(result.metrics["alpha"], 0.1)

    def test_failure_flow_emits_error(self):
        manager = BacktestManager(failing_runner)
        request = BacktestJobRequest(description="fail", parameters={})
        done = threading.Event()
        last_update = {}

        job_id = manager.submit(request)

        def listener(update):
            last_update["value"] = update
            if update.status in (BacktestStatus.SUCCEEDED, BacktestStatus.FAILED):
                done.set()

        manager.subscribe(job_id, listener)
        self.assertTrue(done.wait(timeout=1.0))

        update = last_update["value"]
        self.assertEqual(update.status, BacktestStatus.FAILED)
        self.assertIn("boom", update.error)

        with self.assertRaises(KeyError):
            manager.get_result("nope")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
