from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import get_settings
from routes.api_routes import router as api_router
from routes.chat_routes import router as chat_router
from utils.rag.app_state import initialize_app_state

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from utils.rag.logging_config import setup_logging
    setup_logging()
    initialize_app_state()
    yield


app = FastAPI(title="Skin Cancer RAG Chatbot", lifespan=lifespan)
app.include_router(api_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
