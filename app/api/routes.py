import asyncio
import os
from tempfile import mkstemp

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pdfminer.pdfexceptions import PDFException

from app.core.limiter import ServiceClosedError

CHUNK_SIZE = 1 << 20

router = APIRouter()


@router.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/extract")
async def extract(request: Request, file: UploadFile) -> dict[str, list[dict[str, int | str]]]:
    path: str | None = None
    queued = False
    fd: int | None = None
    try:
        fd, path = mkstemp(dir=request.app.state.config.temp_dir, suffix=".pdf")
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            raise HTTPException(status_code=400, detail="empty file")
        if not chunk.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="invalid pdf")
        os.write(fd, chunk)
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            os.write(fd, chunk)
        os.close(fd)
        fd = None
        future = request.app.state.limiter.enqueue(path)
        queued = True
        pages = await future
        return {"pages": pages}
    except asyncio.QueueFull as exc:
        raise HTTPException(status_code=429, detail="queue full") from exc
    except PDFException as exc:
        raise HTTPException(status_code=400, detail="invalid pdf") from exc
    except ServiceClosedError as exc:
        raise HTTPException(status_code=503, detail="service unavailable") from exc
    finally:
        await file.close()
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if path is not None and not queued:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
