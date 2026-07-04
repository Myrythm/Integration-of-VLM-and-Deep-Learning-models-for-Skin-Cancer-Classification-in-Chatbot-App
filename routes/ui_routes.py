from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from config import PROJECT_ROOT

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> HTMLResponse:
    # chat.html reads its URL params in JS; no server-side template context needed.
    return templates.TemplateResponse("chat.html", {"request": request})
