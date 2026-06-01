from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from app.api.routes import router
from app.core.config import get_config
from app.core.limiter import ExtractionLimiter


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    limiter = ExtractionLimiter(config.parse_concurrency, config.queue_size)
    app.state.config = config
    app.state.limiter = limiter
    await limiter.start()
    try:
        yield
    finally:
        await limiter.stop()


app = FastAPI(
    default_response_class=ORJSONResponse,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)
app.include_router(router)
