from contextlib import asynccontextmanager
import asyncio
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import get_config
from app.core.limiter import ExtractionLimiter


# ---------- Lifespan ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()

    limiter = ExtractionLimiter(
        config.max_workers,
        config.queue_size,
        config.cpu_fallback,
        config.target_latency,
        config.small_file_priority_mb,
        config.medium_file_priority_mb,
    )

    app.state.config = config
    app.state.limiter = limiter

    # safe startup
    try:
        await limiter.start()
    except Exception:
        # fail fast — do not start app in broken state
        raise

    try:
        yield
    finally:
        # graceful shutdown with timeout protection
        try:
            await asyncio.wait_for(limiter.stop(), timeout=10)
        except asyncio.TimeoutError:
            pass


# ---------- App ----------

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    header_name = request.app.state.config.request_id_header
    request_id = request.headers.get(header_name) or uuid4().hex
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers.setdefault(header_name, request_id)
    return response


# ---------- Exception Handlers ----------

@app.exception_handler(Exception)
async def internal_error(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": "internal"},
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_error(_: Request, __: asyncio.TimeoutError) -> JSONResponse:
    return JSONResponse(
        status_code=504,
        content={"error": "timeout"},
    )


# ---------- Routes ----------

app.include_router(router)