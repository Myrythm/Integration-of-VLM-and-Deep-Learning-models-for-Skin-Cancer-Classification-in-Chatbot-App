import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PROJECT_ROOT, get_settings
from routes.api_routes import router as api_router
from routes.chat_routes import router as chat_router
from routes.ui_routes import router as ui_router
from services.rag.app_state import get_chain, initialize_app_state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from services.rag.logging_config import setup_logging

    setup_logging()
    initialize_app_state()
    # Warm the Keras model off the event loop so the first upload doesn't absorb the
    # multi-second load latency. Tolerate environments without the model file (CI).
    try:
        from services.image.classifier import _get_model

        await asyncio.to_thread(_get_model, get_settings().model_path)
    except Exception:
        logger.warning("Model warm-up skipped (model file unavailable)", exc_info=True)
    yield


app = FastAPI(title="Skin Cancer RAG Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
app.include_router(ui_router)
app.include_router(api_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    """Liveness probe: the process is up. Does not assert app state is ready."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness probe: app state (RAG chain) is initialized."""
    try:
        get_chain()
    except RuntimeError:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return JSONResponse(content={"status": "ready"})
