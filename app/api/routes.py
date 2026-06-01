import asyncio
import os
from tempfile import mkstemp
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile

from app.core.limiter import ServiceClosedError
from app.services.extractor import InvalidPDFError

CHUNK_SIZE = 1 << 20  # 1MB

router = APIRouter()


async def _write_upload(file: UploadFile, path: str, max_size: int) -> int:
    size = 0
    first_chunk = True

    with open(path, "wb") as target:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break

            if first_chunk:
                first_chunk = False
                if not chunk.startswith(b"%PDF-"):
                    raise InvalidPDFError()

            size += len(chunk)
            if size > max_size:
                raise HTTPException(413, "file too large")

            await asyncio.to_thread(target.write, chunk)

    if size == 0:
        raise HTTPException(400, "empty file")

    return size


@router.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    config = request.app.state.config
    limiter = request.app.state.limiter
    return {
        "status": "ok",
        "queue": limiter.snapshot(),
        "config": {
            "max_workers": config.max_workers,
            "queue_size": config.queue_size,
            "target_latency": config.target_latency,
            "small_file_priority_mb": config.small_file_priority_mb,
            "medium_file_priority_mb": config.medium_file_priority_mb,
        },
    }


@router.post("/extract")
async def extract(request: Request, response: Response, file: UploadFile) -> dict[str, Any]:
    path: str | None = None
    queued = False
    job_id: str | None = None
    priority_label: str | None = None

    config = request.app.state.config
    limiter = request.app.state.limiter
    request_id = getattr(request.state, "request_id", "")

    try:
        fd, path = mkstemp(dir=config.temp_dir, suffix=".pdf")
        os.close(fd)

        size = await _write_upload(file, path, config.max_file_size)

        size_mb = size / (1024 * 1024)

        timeout = min(
            config.base_job_timeout + (config.timeout_per_mb * size_mb),
            config.max_job_timeout,
        )

        job_id, priority_label, future = await limiter.enqueue(path, timeout, size, request_id)
        queued = True

        try:
            pages = await future
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "timeout") from exc
        except InvalidPDFError as exc:
            raise HTTPException(400, "invalid pdf") from exc

        response.headers[config.request_id_header] = request_id
        meta = pages.setdefault("meta", {})
        meta["request"] = {
            "request_id": request_id,
            "job_id": job_id,
            "priority": priority_label,
            "timeout_seconds": round(timeout, 2),
            "size_bytes": size,
        }
        meta["runtime"] = limiter.snapshot()

        return pages

    except InvalidPDFError as exc:
        raise HTTPException(400, "invalid pdf") from exc

    except asyncio.QueueFull as exc:
        retry_after = limiter.estimate_retry_after(config.retry_after_floor)
        raise HTTPException(429, "queue full", headers={"Retry-After": str(retry_after)}) from exc

    except ServiceClosedError as exc:
        raise HTTPException(503, "service unavailable") from exc

    finally:
        await file.close()

        # cleanup if not handed to worker
        if path is not None and not queued:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass