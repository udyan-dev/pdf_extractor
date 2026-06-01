import asyncio
import os

from app.services.extractor import extract_pdf


class ServiceClosedError(RuntimeError):
    __slots__ = ()


class Job:
    __slots__ = ("path", "future")

    def __init__(self, path: str, future: asyncio.Future[list[dict[str, int | str]]]) -> None:
        self.path = path
        self.future = future


class ExtractionLimiter:
    __slots__ = ("_queue", "_workers", "_closed", "_concurrency")

    def __init__(self, concurrency: int, queue_size: int) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=queue_size)
        self._workers: list[asyncio.Task[None]] = []
        self._closed = False
        self._concurrency = concurrency

    async def start(self) -> None:
        if self._workers:
            return
        for _ in range(self._concurrency):
            self._workers.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self._closed = True
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

    def enqueue(self, path: str) -> asyncio.Future[list[dict[str, int | str]]]:
        if self._closed:
            raise ServiceClosedError("closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[dict[str, int | str]]] = loop.create_future()
        self._queue.put_nowait(Job(path, future))
        return future

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                result = await extract_pdf(job.path)
                if not job.future.cancelled():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.cancelled():
                    job.future.set_exception(exc)
            finally:
                try:
                    os.unlink(job.path)
                except FileNotFoundError:
                    pass
                self._queue.task_done()
