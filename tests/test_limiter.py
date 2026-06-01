import os
import tempfile
import unittest
from unittest.mock import patch

from app.core.limiter import ExtractionLimiter


def _result(label: str) -> dict[str, object]:
    return {
        "meta": {"label": label},
        "pages": [],
        "blocks": [],
        "hierarchy": [],
        "relationships": [],
        "confidence": {"overall": 1.0, "per_block": {}},
        "validation": {"passed": True, "issues": []},
        "layout": {"blocks": []},
        "structure": {"sections": []},
        "raw_text": label,
    }


class ExtractionLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._paths: list[str] = []

    async def asyncTearDown(self) -> None:
        for path in self._paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _temp_pdf(self, size_bytes: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        with open(path, "wb") as handle:
            handle.write(b"%PDF-1.7\n")
            handle.write(b"0" * max(0, size_bytes - 9))
        self._paths.append(path)
        return path

    async def test_enqueue_prioritizes_small_jobs(self) -> None:
        limiter = ExtractionLimiter(
            max_workers=1,
            queue_size=8,
            cpu_fallback=1,
            target_latency=1.0,
            small_file_priority_mb=1.0,
            medium_file_priority_mb=4.0,
        )
        order: list[str] = []

        def fake_extract(path: str) -> dict[str, object]:
            order.append(os.path.basename(path))
            return _result(os.path.basename(path))

        large_path = self._temp_pdf(3 * 1024 * 1024)
        small_path = self._temp_pdf(128 * 1024)

        with patch("app.core.limiter.extract_pdf", side_effect=fake_extract):
            _, _, large_future = await limiter.enqueue(large_path, 5.0, 3 * 1024 * 1024, "req-large")
            _, _, small_future = await limiter.enqueue(small_path, 5.0, 128 * 1024, "req-small")
            await limiter.start()
            await large_future
            await small_future
            await limiter.stop()

        self.assertEqual(order[0], os.path.basename(small_path))
        self.assertEqual(order[1], os.path.basename(large_path))

    async def test_snapshot_tracks_completed_work(self) -> None:
        limiter = ExtractionLimiter(
            max_workers=1,
            queue_size=8,
            cpu_fallback=1,
            target_latency=1.0,
            small_file_priority_mb=1.0,
            medium_file_priority_mb=4.0,
        )
        path = self._temp_pdf(128 * 1024)

        with patch("app.core.limiter.extract_pdf", return_value=_result("done")):
            _, _, future = await limiter.enqueue(path, 5.0, 128 * 1024, "req-1")
            await limiter.start()
            await future
            snapshot = limiter.snapshot()
            await limiter.stop()

        self.assertEqual(snapshot["completed_jobs"], 1)
        self.assertEqual(snapshot["failed_jobs"], 0)
        self.assertGreaterEqual(snapshot["peak_queue_depth"], 1)
        self.assertIn("target_concurrency", snapshot)