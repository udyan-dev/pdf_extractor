import asyncio
import math
import os
from uuid import uuid4
from typing import Any

from app.services.extractor import InvalidPDFError, extract_pdf


class ServiceClosedError(RuntimeError):
    __slots__ = ()


class Job:
    __slots__ = ("id", "path", "future", "timeout", "enqueued_at", "size_bytes", "priority", "priority_label", "request_id")

    def __init__(
        self,
        job_id: str,
        path: str,
        future: asyncio.Future[dict[str, Any]],
        timeout: float,
        enqueued_at: float,
        size_bytes: int,
        priority: int,
        priority_label: str,
        request_id: str,
    ) -> None:
        self.id = job_id
        self.path = path
        self.future = future
        self.timeout = timeout
        self.enqueued_at = enqueued_at
        self.size_bytes = size_bytes
        self.priority = priority
        self.priority_label = priority_label
        self.request_id = request_id


class ExtractionLimiter:
    __slots__ = (
        "_queue",
        "_workers",
        "_closed",
        "_max_workers",
        "_target_concurrency",
        "_active_jobs",
        "_gate",
        "_small_file_priority_bytes",
        "_medium_file_priority_bytes",
        "_avg_latency",
        "_avg_queue_latency",
        "_failure_rate",
        "_target_latency",
        "_completed_jobs",
        "_failed_jobs",
        "_peak_queue_depth",
        "_job_sequence",
    )

    def __init__(
        self,
        max_workers: int,
        queue_size: int,
        cpu_fallback: int,
        target_latency: float,
        small_file_priority_mb: float,
        medium_file_priority_mb: float,
    ) -> None:
        cpu_count = os.cpu_count() or cpu_fallback

        self._max_workers = min(max_workers, cpu_count)
        self._target_concurrency = max(1, min(self._max_workers, max(1, math.ceil(self._max_workers / 2))))
        self._small_file_priority_bytes = int(small_file_priority_mb * 1024 * 1024)
        self._medium_file_priority_bytes = int(medium_file_priority_mb * 1024 * 1024)

        self._queue: asyncio.PriorityQueue[tuple[int, int, Job]] = asyncio.PriorityQueue(maxsize=queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False

        self._active_jobs = 0
        self._gate = asyncio.Condition()

        # metrics (EMA)
        self._avg_latency = 0.0
        self._avg_queue_latency = 0.0
        self._failure_rate = 0.0

        self._target_latency = target_latency
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._peak_queue_depth = 0
        self._job_sequence = 0

    # ---------- Lifecycle ----------

    async def start(self) -> None:
        if self._workers:
            return

        for _ in range(self._max_workers):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self._closed = True
        await self._queue.join()

        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

    # ---------- Public API ----------

    async def enqueue(
        self,
        path: str,
        timeout: float,
        size_bytes: int,
        request_id: str,
    ) -> tuple[str, str, asyncio.Future[dict[str, Any]]]:
        if self._closed:
            raise ServiceClosedError("closed")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        priority, priority_label = self._priority_for_size(size_bytes)
        job = Job(
            job_id=uuid4().hex,
            path=path,
            future=future,
            timeout=timeout,
            enqueued_at=loop.time(),
            size_bytes=size_bytes,
            priority=priority,
            priority_label=priority_label,
            request_id=request_id,
        )

        self._queue.put_nowait((priority, self._job_sequence, job))
        self._job_sequence += 1
        self._peak_queue_depth = max(self._peak_queue_depth, self._queue.qsize())

        # lightweight retune (no locks)
        self._retune()

        return job.id, priority_label, future

    # ---------- Worker ----------

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()

        while True:
            _, _, job = await self._queue.get()

            started_at = 0.0
            acquired = False

            try:
                await self._acquire_slot()
                acquired = True
                started_at = loop.time()

                # queue latency
                self._avg_queue_latency = self._ema(
                    self._avg_queue_latency,
                    started_at - job.enqueued_at,
                )

                result = await asyncio.wait_for(
                    asyncio.to_thread(extract_pdf, job.path),
                    timeout=job.timeout,
                )

                if not result["raw_text"].strip():
                    raise InvalidPDFError()

                latency = loop.time() - started_at
                self._avg_latency = self._ema(self._avg_latency, latency)
                self._failure_rate = self._ema(self._failure_rate, 0.0)

                if not job.future.cancelled():
                    job.future.set_result(result)
                self._completed_jobs += 1

            except Exception as exc:
                if started_at:
                    latency = loop.time() - started_at
                    self._avg_latency = self._ema(self._avg_latency, latency)

                if not isinstance(exc, InvalidPDFError):
                    self._failure_rate = self._ema(self._failure_rate, 1.0)
                else:
                    self._failure_rate = self._ema(self._failure_rate, 0.0)

                if not job.future.cancelled():
                    job.future.set_exception(exc)
                self._failed_jobs += 1

            finally:
                if acquired:
                    await self._release_slot()

                try:
                    os.unlink(job.path)
                except FileNotFoundError:
                    pass

                self._queue.task_done()
                self._retune()

    # ---------- Adaptive Concurrency ----------

    def _retune(self) -> None:
        self._target_concurrency = self._compute_target()

    def _compute_target(self) -> int:
        target = self._target_concurrency
        pending = self._queue.qsize()

        # scale up (slow)
        if (
            pending > target
            and self._avg_latency < self._target_latency
            and self._failure_rate < 0.2
        ):
            target += 1

        if pending > target * 2 and self._failure_rate < 0.15:
            target += 1

        # scale down (fast protection)
        if (
            self._avg_latency > self._target_latency * 1.5
            or self._failure_rate > 0.25
        ):
            target = max(1, target // 2)

        if target < 1:
            return 1
        if target > self._max_workers:
            return self._max_workers

        return target

    def snapshot(self) -> dict[str, Any]:
        return {
            "closed": self._closed,
            "queue_depth": self._queue.qsize(),
            "peak_queue_depth": self._peak_queue_depth,
            "active_jobs": self._active_jobs,
            "max_workers": self._max_workers,
            "target_concurrency": self._target_concurrency,
            "completed_jobs": self._completed_jobs,
            "failed_jobs": self._failed_jobs,
            "avg_latency_seconds": round(self._avg_latency, 4),
            "avg_queue_latency_seconds": round(self._avg_queue_latency, 4),
            "failure_rate": round(self._failure_rate, 4),
        }

    def estimate_retry_after(self, floor_seconds: float) -> int:
        base_latency = self._avg_latency or self._target_latency or floor_seconds
        queue_depth = max(self._queue.qsize(), 1)
        concurrency = max(self._target_concurrency, 1)
        estimated = max(floor_seconds, (queue_depth / concurrency) * base_latency)
        return max(1, math.ceil(estimated))

    def _priority_for_size(self, size_bytes: int) -> tuple[int, str]:
        if size_bytes <= self._small_file_priority_bytes:
            return 0, "small"
        if size_bytes <= self._medium_file_priority_bytes:
            return 1, "medium"
        return 2, "large"

    async def _acquire_slot(self) -> None:
        async with self._gate:
            while self._active_jobs >= self._target_concurrency:
                await self._gate.wait()

            self._active_jobs += 1

    async def _release_slot(self) -> None:
        async with self._gate:
            self._active_jobs -= 1
            self._gate.notify_all()

    # ---------- Utils ----------

    @staticmethod
    def _ema(current: float, value: float) -> float:
        if current == 0.0:
            return value
        return (current * 0.8) + (value * 0.2)