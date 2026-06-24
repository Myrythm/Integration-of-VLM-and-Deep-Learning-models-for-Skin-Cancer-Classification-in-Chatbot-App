# UI + Folder Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3-page UI (Home, Upload, Chat) with dark mode to the v2 RAG skin cancer chatbot, and restructure `utils/` → `services/` for descriptive folder naming.

**Architecture:** Jinja2 templates + Tailwind CDN + vanilla JS. Clinical design (medical teal, warm neutral, amber warnings) with dark mode toggle. `utils/` renamed to `services/` with `services/image/` and `services/rag/` subfolders. All existing 64 tests must pass after refactor.

**Tech Stack:** FastAPI + Jinja2 (starlette), Tailwind CSS CDN, vanilla JS (EventSource/fetch), Inter font (Google Fonts), pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-20-ui-and-folder-restructure-design.md`

---

## Global Constraints

- **Python**: 3.10+ (venv already created with 3.10.20 via uv)
- **Environment**: `source .venv/bin/activate` before running any Python/pytest commands
- **Dependencies**: Use `uv pip install -r requirements.txt` (not raw pip)
- **Imports**: Absolute imports only (`from services.rag.x import Y`)
- **No comments** in code unless user explicitly asks
- **Type hints**: Required on all Python function signatures
- **Commit format**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `chore:`, `test:`, `docs:`)
- **One commit per task**
- **Branch**: `feat/ui-and-restructure` (create from current `feat/integrate-efficientnet-model` or `main`)
- **Dark mode**: Tailwind `darkMode: 'class'` strategy, toggle via `<html class="dark">`
- **UI language**: Bahasa Indonesia for all chrome (navbar, buttons, labels, disclaimers)
- **Color palette**: Teal `#0d9488` primary, amber `#d97706` accent, warm white `#fafaf9` bg (light), slate-900 `#0f172a` (dark)
- **No AI slop**: No gradients, no "AI-powered" copy, no generic SaaS templates, no glassmorphism

---

## File Structure

```
project-root/
├── services/                        # NEW (replaces utils/)
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   └── classifier.py            # moved from utils/image_classifier.py
│   └── rag/                         # moved from utils/rag/
│       └── (17 files, same names)
│
├── routes/
│   ├── api_routes.py                # update imports
│   ├── chat_routes.py              # update imports
│   └── ui_routes.py                # NEW
│
├── templates/                       # NEW
│   ├── base.html
│   ├── home.html
│   ├── upload.html
│   ├── chat.html
│   └── partials/
│       ├── navbar.html
│       └── disclaimer_bar.html
│
├── static/                          # NEW
│   ├── css/
│   │   └── app.css
│   └── js/
│       ├── theme.js
│       ├── upload.js
│       └── chat.js
│
├── tests/
│   └── integration/
│       └── test_ui_routes.py        # NEW
```

---

## Task 1: Refactor `utils/` → `services/`

**Files:**
- Move: `utils/image_classifier.py` → `services/image/classifier.py`
- Move: `utils/rag/*.py` (17 files) → `services/rag/*.py`
- Move: `utils/__init__.py` → `services/__init__.py`
- Move: `utils/rag/__init__.py` → `services/rag/__init__.py`
- Create: `services/image/__init__.py` (empty)
- Modify: ~30 files (import updates: `from utils.` → `from services.`)
- Delete: `utils/` directory (after move)

**Interfaces:**
- Consumes: nothing (first task, mechanical refactor)
- Produces: all modules accessible via `from services.rag.x import Y` and `from services.image.classifier import Z`

- [ ] **Step 1: Clean pycache and create services/ structure**

```bash
find . -name __pycache__ -type d -not -path "./.venv/*" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null
mkdir -p services/image
touch services/image/__init__.py
```

- [ ] **Step 2: git mv all files from utils/ to services/**

```bash
git mv utils/__init__.py services/__init__.py
git mv utils/rag/__init__.py services/rag/__init__.py
git mv utils/rag/app_state.py services/rag/app_state.py
git mv utils/rag/cache.py services/rag/cache.py
git mv utils/rag/chain.py services/rag/chain.py
git mv utils/rag/citation.py services/rag/citation.py
git mv utils/rag/disclaimer.py services/rag/disclaimer.py
git mv utils/rag/embedder.py services/rag/embedder.py
git mv utils/rag/ingestion.py services/rag/ingestion.py
git mv utils/rag/language.py services/rag/language.py
git mv utils/rag/llm_provider.py services/rag/llm_provider.py
git mv utils/rag/logging_config.py services/rag/logging_config.py
git mv utils/rag/memory.py services/rag/memory.py
git mv utils/rag/prompt.py services/rag/prompt.py
git mv utils/rag/pubmed.py services/rag/pubmed.py
git mv utils/rag/retriever.py services/rag/retriever.py
git mv utils/rag/safety.py services/rag/safety.py
git mv utils/rag/vector_store.py services/rag/vector_store.py
git mv utils/image_classifier.py services/image/classifier.py
```

- [ ] **Step 3: Update all imports — `from utils.rag` → `from services.rag`**

Find all files with `from utils` imports:

```bash
grep -rl "from utils\." --include="*.py" . | grep -v ".venv"
```

For each file found, replace:
- `from utils.rag.` → `from services.rag.`
- `from utils.image_classifier` → `from services.image.classifier`

Use sed for bulk replacement:

```bash
find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" -exec sed -i 's/from utils\.rag\./from services.rag./g' {} +
find . -name "*.py" -not -path "./.venv/*" -not -path "./.git/*" -exec sed -i 's/from utils\.image_classifier/from services.image.classifier/g' {} +
```

- [ ] **Step 4: Verify no remaining `from utils` imports**

```bash
grep -r "from utils\." --include="*.py" . | grep -v ".venv"
```

Expected: no output (0 results).

- [ ] **Step 5: Delete empty utils/ directory**

```bash
rm -rf utils/
```

- [ ] **Step 6: Run all tests to verify refactor**

```bash
source .venv/bin/activate
pytest -q --ignore=tests/integration/test_e2e_smoke.py
```

Expected: 64 passed, 1 skipped.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: rename utils/ to services/ for descriptive folder structure

Move image_classifier.py to services/image/classifier.py and all rag
modules to services/rag/. Update ~30 files' imports. No logic changes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Task 2: Create UI Routes + Tests + main.py Update

**Files:**
- Create: `routes/ui_routes.py`
- Create: `tests/integration/test_ui_routes.py`
- Modify: `main.py` (re-add StaticFiles, Jinja2Templates, register ui_router)

**Interfaces:**
- Consumes: `main.py:app` (from Task 1)
- Produces:
  - `GET /` → renders `home.html`
  - `GET /upload` → renders `upload.html`
  - `GET /chat` → renders `chat.html` (with optional query params: session, label, confidence)
  - `router` (APIRouter) at `routes/ui_routes.py:router`

- [ ] **Step 1: Write failing tests `tests/integration/test_ui_routes.py`**

```python
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home_page_returns_200_with_content() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Skrining Kanker Kulit" in response.text


def test_upload_page_returns_200_with_content() -> None:
    response = client.get("/upload")
    assert response.status_code == 200
    assert "Upload Foto Lesi" in response.text


def test_chat_page_returns_200_with_content() -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "Konteks Klasifikasi" in response.text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source .venv/bin/activate
pytest tests/integration/test_ui_routes.py -v
```

Expected: 3 FAILED (404 — routes don't exist yet).

- [ ] **Step 3: Create `routes/ui_routes.py`**

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")

router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("home.html", {"request": request})


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    session: str | None = None,
    label: str | None = None,
    confidence: float | None = None,
) -> HTMLResponse:
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "session": session or "",
            "label": label or "",
            "confidence": confidence or 0.0,
        },
    )
```

- [ ] **Step 4: Update `main.py` to register ui_router and mount static**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import get_settings
from routes.api_routes import router as api_router
from routes.chat_routes import router as chat_router
from routes.ui_routes import router as ui_router
from services.rag.app_state import initialize_app_state

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from services.rag.logging_config import setup_logging
    setup_logging()
    initialize_app_state()
    yield


app = FastAPI(title="Skin Cancer RAG Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(ui_router)
app.include_router(api_router)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
```

- [ ] **Step 5: Create minimal templates so tests can pass**

Create `templates/base.html`:

```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SkinVision{% endblock %}</title>
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

Create `templates/home.html`:

```html
{% extends "base.html" %}
{% block title %}SkinVision — Skrining Kanker Kulit{% endblock %}
{% block content %}
<h1>Skrining Kanker Kulit</h1>
{% endblock %}
```

Create `templates/upload.html`:

```html
{% extends "base.html" %}
{% block title %}Upload Foto Lesi — SkinVision{% endblock %}
{% block content %}
<h1>Upload Foto Lesi</h1>
{% endblock %}
```

Create `templates/chat.html`:

```html
{% extends "base.html" %}
{% block title %}Chat — SkinVision{% endblock %}
{% block content %}
<h1>Konteks Klasifikasi</h1>
{% endblock %}
```

Create `static/css/app.css` (empty for now):

```css
/* Custom styles — populated in Task 3 */
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/integration/test_ui_routes.py -v
```

Expected: 3 PASSED.

- [ ] **Step 7: Run full test suite to check for regressions**

```bash
pytest -q --ignore=tests/integration/test_e2e_smoke.py
```

Expected: 67 passed, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(ui): add UI routes and minimal templates for home, upload, chat

GET /, GET /upload, GET /chat with Jinja2 templates. main.py re-adds
StaticFiles mount and ui_router registration. Minimal templates created
as stubs — full design in subsequent tasks.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Task 3: Create base.html with Tailwind + Dark Mode + Navbar + Partials

**Files:**
- Modify: `templates/base.html` (full implementation)
- Create: `templates/partials/navbar.html`
- Create: `templates/partials/disclaimer_bar.html`
- Create: `static/js/theme.js`
- Modify: `static/css/app.css`

**Interfaces:**
- Consumes: `routes/ui_routes.py` (from Task 2)
- Produces: `base.html` with Tailwind CDN, dark mode config, navbar, `{% block content %}` for child templates

- [ ] **Step 1: Create `static/js/theme.js`**

```javascript
(function () {
    const stored = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    if (stored === "dark" || (!stored && prefersDark)) {
        document.documentElement.classList.add("dark");
    }
})();

function toggleTheme() {
    const html = document.documentElement;
    const isDark = html.classList.toggle("dark");
    localStorage.setItem("theme", isDark ? "dark" : "light");
    const icon = document.getElementById("theme-icon");
    if (icon) {
        icon.innerHTML = isDark
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
    }
}
```

- [ ] **Step 2: Create `static/css/app.css`**

```css
:root {
    --color-bg: #fafaf9;
    --color-surface: #ffffff;
    --color-text: #1e293b;
    --color-text-muted: #64748b;
    --color-primary: #0d9488;
    --color-primary-dark: #0f766e;
    --color-accent: #d97706;
    --color-success: #15803d;
    --color-border: #e2e8f0;
}

html.dark {
    --color-bg: #0f172a;
    --color-surface: #1e293b;
    --color-text: #f1f5f9;
    --color-text-muted: #94a3b8;
    --color-primary: #14b8a6;
    --color-primary-dark: #0d9488;
    --color-accent: #f59e0b;
    --color-success: #22c55e;
    --color-border: #334155;
}

body {
    transition: background-color 0.2s ease, color 0.2s ease;
}
```

- [ ] **Step 3: Create `templates/partials/navbar.html`**

```html
<nav class="border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 sticky top-0 z-50">
    <div class="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
        <a href="/" class="text-lg font-bold tracking-tight text-slate-800 dark:text-slate-100">
            SkinVision
        </a>
        <div class="flex items-center gap-6">
            <a href="/" class="text-sm text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400">Beranda</a>
            <a href="/upload" class="text-sm text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400">Deteksi</a>
            <button onclick="toggleTheme()" class="p-1 text-slate-600 dark:text-slate-300 hover:text-teal-600 dark:hover:text-teal-400" aria-label="Toggle dark mode">
                <span id="theme-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
                </span>
            </button>
        </div>
    </div>
</nav>
<script>
    (function() {
        const isDark = document.documentElement.classList.contains("dark");
        const icon = document.getElementById("theme-icon");
        if (isDark && icon) {
            icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';
        }
    })();
</script>
```

- [ ] **Step 4: Create `templates/partials/disclaimer_bar.html`**

```html
<div class="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-600 dark:border-amber-500 px-4 py-3 text-sm text-amber-800 dark:text-amber-200">
    {% block disclaimer_text %}
    Alat ini untuk edukasi dan skrining awal, bukan diagnosis medis. Hasil AI bisa keliru.
    {% endblock %}
</div>
```

- [ ] **Step 5: Rewrite `templates/base.html` with full Tailwind + dark mode + partials**

```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SkinVision{% endblock %}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>tailwind = { config: { darkMode: 'class' } };</script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="/static/css/app.css">
    <script>
        (function() {
            const stored = localStorage.getItem('theme');
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (stored === 'dark' || (!stored && prefersDark)) {
                document.documentElement.classList.add('dark');
            }
        })();
    </script>
</head>
<body class="bg-stone-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans antialiased transition-colors duration-200">
    {% include "partials/navbar.html" %}
    <main class="max-w-4xl mx-auto px-4 py-8">
        {% block content %}{% endblock %}
    </main>
    <script src="/static/js/theme.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

- [ ] **Step 6: Run UI tests to verify templates still render**

```bash
pytest tests/integration/test_ui_routes.py -v
```

Expected: 3 PASSED (content strings still present in stub templates).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(ui): base.html with Tailwind CDN, dark mode toggle, navbar, partials

Tailwind via CDN with darkMode:'class'. theme.js handles localStorage
persistence and OS preference. Navbar has Beranda/Deteksi links + theme
toggle. Disclaimer bar partial with amber accent.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Task 4: Create `templates/home.html` — Full Clinical Design

**Files:**
- Modify: `templates/home.html`

**Interfaces:**
- Consumes: `templates/base.html` (from Task 3)
- Produces: Home page with clinical content (jenis kanker, kapan ke dokter, disclaimer)

- [ ] **Step 1: Write `templates/home.html`**

```html
{% extends "base.html" %}
{% block title %}SkinVision — Skrining Kanker Kulit{% endblock %}
{% block content %}

<div class="space-y-8">
    <section>
        <h1 class="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
            Skrining Kanker Kulit
        </h1>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-1">
            berbasis EfficientNetB3
        </p>
        <p class="text-sm text-slate-700 dark:text-slate-300 mt-4 leading-relaxed">
            Upload foto lesi kulit untuk klasifikasi 4 jenis: BCC, SCC, Melanoma, Nevus.
            Hasil dilanjutkan ke chatbot edukasi bilingual (Indonesia/Inggris).
        </p>
        <a href="/upload"
           class="inline-block mt-6 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600 text-white text-sm font-medium rounded-lg transition-colors">
            Upload Foto Lesi
        </a>
    </section>

    <section>
        <h2 class="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-3">
            Jenis yang Dideteksi
        </h2>
        <dl class="space-y-3 text-sm">
            <div class="flex gap-3">
                <dt class="font-medium text-slate-800 dark:text-slate-200 min-w-[180px]">Karsinoma Sel Basal</dt>
                <dd class="text-slate-600 dark:text-slate-400">Jenis paling umum, tumbuh lambat</dd>
            </div>
            <div class="flex gap-3">
                <dt class="font-medium text-slate-800 dark:text-slate-200 min-w-[180px]">Karsinoma Sel Skuamosa</dt>
                <dd class="text-slate-600 dark:text-slate-400">Bisa menyebar ke lapisan kulit lebih dalam</dd>
            </div>
            <div class="flex gap-3">
                <dt class="font-medium text-slate-800 dark:text-slate-200 min-w-[180px]">Melanoma</dt>
                <dd class="text-slate-600 dark:text-slate-400">Paling serius, butuh deteksi dini</dd>
            </div>
            <div class="flex gap-3">
                <dt class="font-medium text-slate-800 dark:text-slate-200 min-w-[180px]">Nevus</dt>
                <dd class="text-slate-600 dark:text-slate-400">Tahi lalat jinak, umumnya tidak berbahaya</dd>
            </div>
        </dl>
    </section>

    <section>
        <h2 class="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400 mb-3">
            Kapan Harus ke Dokter
        </h2>
        <ul class="space-y-2 text-sm text-slate-700 dark:text-slate-300 list-disc list-inside">
            <li>Lesi berubah bentuk, warna, atau ukuran</li>
            <li>Berdarah atau gatal tanpa sebab</li>
            <li>Tahi lalat baru setelah usia 30</li>
            <li>Lesi yang tidak sembuh dalam 2 minggu</li>
        </ul>
        <p class="text-sm text-slate-600 dark:text-slate-400 mt-3">
            Konsultasi ke dokter spesialis kulit (Sp.KK) di rumah sakit terdekat.
        </p>
    </section>

    {% include "partials/disclaimer_bar.html" %}

</div>

{% endblock %}
```

- [ ] **Step 2: Run UI test to verify content**

```bash
pytest tests/integration/test_ui_routes.py::test_home_page_returns_200_with_content -v
```

Expected: PASSED ("Skrining Kanker Kulit" is in the page).

- [ ] **Step 3: Commit**

```bash
git add templates/home.html
git commit -m "feat(ui): home page with clinical content (jenis kanker, kapan ke dokter)

Document-style layout. Definition list for cancer types, bullet list for
when to see a doctor, Sp.KK reference, amber disclaimer bar. No AI slop.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Task 5: Create `templates/upload.html` + `static/js/upload.js`

**Files:**
- Modify: `templates/upload.html`
- Create: `static/js/upload.js`

**Interfaces:**
- Consumes: `templates/base.html`, `POST /api/upload` (existing)
- Produces: Upload page with drop zone, tips, preview, result card, link to chat

- [ ] **Step 1: Write `templates/upload.html`**

```html
{% extends "base.html" %}
{% block title %}Upload Foto Lesi — SkinVision{% endblock %}
{% block content %}

<div class="space-y-6">
    <h1 class="text-2xl font-bold tracking-tight text-slate-900 dark:text-slate-100">
        Upload Foto Lesi
    </h1>

    <div id="drop-zone"
         class="border-2 border-dashed border-teal-600 dark:border-teal-500 rounded-lg p-8 text-center cursor-pointer hover:bg-teal-50 dark:hover:bg-teal-900/10 transition-colors">
        <p class="text-sm text-slate-700 dark:text-slate-300">Pilih atau drop foto di sini</p>
        <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Format: PNG, JPG, WEBP — Maksimal: 10 MB</p>
        <button type="button" onclick="document.getElementById('file-input').click()"
                class="mt-4 px-4 py-2 bg-teal-600 hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600 text-white text-sm font-medium rounded-lg transition-colors">
            Pilih File
        </button>
        <input type="file" id="file-input" accept="image/png,image/jpeg,image/webp" class="hidden">
    </div>

    <div id="preview-container" class="hidden">
        <p class="text-sm font-medium text-slate-500 dark:text-slate-400 mb-2">Preview</p>
        <img id="preview-img" class="max-h-48 rounded-lg border border-slate-200 dark:border-slate-700" alt="Preview">
    </div>

    <div class="text-sm text-slate-700 dark:text-slate-300">
        <p class="font-medium mb-2">Tips Foto yang Baik:</p>
        <ul class="space-y-1 list-disc list-inside text-slate-600 dark:text-slate-400">
            <li>Pencahayaan terang, tidak silau</li>
            <li>Fokus pada lesi, background netral</li>
            <li>Jarak 10-15 cm, foto tajam</li>
            <li>Satu lesi per foto</li>
        </ul>
    </div>

    <button id="classify-btn" disabled
            class="px-5 py-2.5 bg-teal-600 hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
        Klasifikasikan
    </button>

    <div id="result" class="hidden space-y-3">
        <h2 class="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Hasil Klasifikasi
        </h2>
        <div class="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 space-y-2">
            <div class="flex justify-between items-center">
                <span class="text-sm font-medium text-slate-700 dark:text-slate-300">Diagnosis</span>
                <span id="result-label" class="text-sm font-bold text-slate-900 dark:text-slate-100"></span>
            </div>
            <div class="flex justify-between items-center">
                <span class="text-sm font-medium text-slate-700 dark:text-slate-300">Confidence</span>
                <div class="flex items-center gap-2">
                    <div class="w-32 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                        <div id="result-bar" class="bg-teal-600 dark:bg-teal-500 h-2 rounded-full transition-all" style="width:0%"></div>
                    </div>
                    <span id="result-confidence" class="text-sm font-mono text-slate-700 dark:text-slate-300"></span>
                    <span id="result-level" class="text-xs font-medium text-slate-500 dark:text-slate-400"></span>
                </div>
            </div>
            <p class="text-xs font-mono text-slate-400 dark:text-slate-500" id="result-model"></p>
        </div>
        <div class="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-600 dark:border-amber-500 px-4 py-2 text-sm text-amber-800 dark:text-amber-200">
            Hasil AI bisa keliru. Konfirmasi dengan dokter spesialis kulit.
        </div>
        <a id="chat-link" href="#"
           class="inline-block px-5 py-2.5 bg-teal-600 hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600 text-white text-sm font-medium rounded-lg transition-colors">
            Konsultasi Chatbot
        </a>
    </div>
</div>

{% endblock %}

{% block scripts %}
<script src="/static/js/upload.js"></script>
{% endblock %}
```

- [ ] **Step 2: Write `static/js/upload.js`**

```javascript
(function () {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewContainer = document.getElementById("preview-container");
    const previewImg = document.getElementById("preview-img");
    const classifyBtn = document.getElementById("classify-btn");
    const resultDiv = document.getElementById("result");
    let selectedFile = null;

    function handleFile(file) {
        if (!file) return;
        if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
            alert("Format tidak didukung. Gunakan PNG, JPG, atau WEBP.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert("File terlalu besar. Maksimal 10 MB.");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
        classifyBtn.disabled = false;
    }

    fileInput.addEventListener("change", function (e) {
        handleFile(e.target.files[0]);
    });

    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        dropZone.classList.add("bg-teal-50", "dark:bg-teal-900/10");
    });

    dropZone.addEventListener("dragleave", function () {
        dropZone.classList.remove("bg-teal-50", "dark:bg-teal-900/10");
    });

    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("bg-teal-50", "dark:bg-teal-900/10");
        handleFile(e.dataTransfer.files[0]);
    });

    classifyBtn.addEventListener("click", async function () {
        if (!selectedFile) return;
        classifyBtn.disabled = true;
        classifyBtn.textContent = "Memproses...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Gagal mengklasifikasi");
            }
            const data = await response.json();
            const det = data.detection;
            const conf = (det.confidence * 100).toFixed(2);
            const level = det.confidence >= 0.8 ? "Tinggi" : det.confidence >= 0.5 ? "Sedang" : "Rendah";
            const levelColor = det.confidence >= 0.8 ? "text-green-700 dark:text-green-400" : det.confidence >= 0.5 ? "text-amber-700 dark:text-amber-400" : "text-red-700 dark:text-red-400";

            document.getElementById("result-label").textContent = det.label;
            document.getElementById("result-confidence").textContent = conf + "%";
            document.getElementById("result-bar").style.width = conf + "%";
            const levelEl = document.getElementById("result-level");
            levelEl.textContent = level;
            levelEl.className = "text-xs font-medium " + levelColor;
            document.getElementById("result-model").textContent = "Model: " + det.model_version;

            const chatUrl = "/chat?session=" + encodeURIComponent(data.chat_session_id) +
                "&label=" + encodeURIComponent(det.label) +
                "&confidence=" + encodeURIComponent(det.confidence);
            document.getElementById("chat-link").href = chatUrl;

            resultDiv.classList.remove("hidden");
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            classifyBtn.disabled = false;
            classifyBtn.textContent = "Klasifikasikan";
        }
    });
})();
```

- [ ] **Step 3: Run UI test to verify content**

```bash
pytest tests/integration/test_ui_routes.py::test_upload_page_returns_200_with_content -v
```

Expected: PASSED ("Upload Foto Lesi" is in the page).

- [ ] **Step 4: Commit**

```bash
git add templates/upload.html static/js/upload.js
git commit -m "feat(ui): upload page with drop zone, tips, result card, chat link

Medical intake form layout. Drag-drop zone with file validation,
photo tips, preview, result table with confidence bar + level
label (Tinggi/Sedang/Rendah), inline amber warning, link to chat
with session/label/confidence params. upload.js handles fetch POST.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Task 6: Create `templates/chat.html` + `static/js/chat.js`

**Files:**
- Modify: `templates/chat.html`
- Create: `static/js/chat.js`

**Interfaces:**
- Consumes: `templates/base.html`, `POST /api/chat` (existing, SSE)
- Produces: Chat page with context header, chat area, SSE streaming, citation list, disclaimer

- [ ] **Step 1: Write `templates/chat.html`**

```html
{% extends "base.html" %}
{% block title %}Chat — SkinVision{% endblock %}
{% block content %}

<div class="space-y-4">
    <a href="/upload" class="text-sm text-slate-600 dark:text-slate-400 hover:text-teal-600 dark:hover:text-teal-400">
        ← Kembali ke Upload
    </a>

    <div class="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-3">
        <p class="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Konteks Klasifikasi</p>
        <p class="text-sm font-mono text-slate-700 dark:text-slate-300 mt-1">
            <span id="ctx-label">{{ label }}</span> · <span id="ctx-confidence">{{ (confidence * 100)|round(2) }}%</span> · EfficientNetB3-v1
        </p>
    </div>

    <div id="chat-area" class="space-y-4 min-h-[300px] max-h-[60vh] overflow-y-auto pb-4">
    </div>

    <div class="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-600 dark:border-amber-500 px-4 py-2 text-sm text-amber-800 dark:text-amber-200">
        ⚠ Asisten AI untuk edukasi. Bukan nasihat medis. Konsultasi dokter spesialis kulit (Sp.KK).
    </div>

    <div class="flex gap-2">
        <textarea id="chat-input" rows="2" placeholder="Ketik pesan..."
                  class="flex-1 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-slate-800 dark:text-slate-200 focus:outline-none focus:border-teal-600 dark:focus:border-teal-500 resize-none"></textarea>
        <button id="send-btn" onclick="sendMessage()"
                class="px-4 py-2 bg-teal-600 hover:bg-teal-700 dark:bg-teal-500 dark:hover:bg-teal-600 text-white text-sm font-medium rounded-lg transition-colors">
            →
        </button>
    </div>
</div>

{% endblock %}

{% block scripts %}
<script>
    window.CHAT_SESSION = "{{ session }}";
    window.CHAT_LABEL = "{{ label }}";
    window.CHAT_CONFIDENCE = {{ confidence }};
</script>
<script src="/static/js/chat.js"></script>
{% endblock %}
```

- [ ] **Step 2: Write `static/js/chat.js`**

```javascript
(function () {
    const chatArea = document.getElementById("chat-area");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    let isStreaming = false;

    chatInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    window.sendMessage = async function () {
        const message = chatInput.value.trim();
        if (!message || isStreaming) return;

        isStreaming = true;
        sendBtn.disabled = true;
        chatInput.value = "";

        const userBubble = document.createElement("div");
        userBubble.className = "flex justify-end";
        userBubble.innerHTML = '<div class="max-w-[80%] bg-teal-50 dark:bg-teal-900/20 rounded-lg px-4 py-2 text-sm"><p class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">User</p><p class="text-slate-800 dark:text-slate-200"></p></div>';
        userBubble.querySelector("p:last-child").textContent = message;
        chatArea.appendChild(userBubble);

        const assistantBubble = document.createElement("div");
        assistantBubble.className = "flex justify-start";
        assistantBubble.innerHTML = '<div class="max-w-[80%] bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg px-4 py-2 text-sm w-full"><p class="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Asisten</p><div id="assistant-text" class="text-slate-800 dark:text-slate-200"></div><div id="assistant-citations" class="mt-2 space-y-1"></div></div>';
        chatArea.appendChild(assistantBubble);

        const textDiv = assistantBubble.querySelector("#assistant-text");
        const citationDiv = assistantBubble.querySelector("#assistant-citations");
        textDiv.innerHTML = "&#9608;";

        chatArea.scrollTop = chatArea.scrollHeight;

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: window.CHAT_SESSION,
                    message: message,
                    detection: {
                        label: window.CHAT_LABEL,
                        confidence: window.CHAT_CONFIDENCE,
                        model_version: "EfficientNetB3-v1",
                    },
                }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const data = JSON.parse(line.slice(6));

                    if (data.type === "token") {
                        fullText += data.content || "";
                        textDiv.textContent = fullText + "\u2588";
                        chatArea.scrollTop = chatArea.scrollHeight;
                    } else if (data.type === "citation" && data.citations) {
                        let html = '<p class="text-xs font-medium text-slate-500 dark:text-slate-400">Referensi:</p>';
                        for (const c of data.citations) {
                            html += '<p class="text-xs"><a href="' + c.url + '" target="_blank" class="text-teal-600 dark:text-teal-400 hover:underline">[' + c.number + '] ' + c.title + ' — ' + c.source + '</a></p>';
                        }
                        citationDiv.innerHTML = html;
                    } else if (data.type === "blocked") {
                        textDiv.textContent = data.content || "";
                    } else if (data.type === "done") {
                        textDiv.textContent = fullText;
                    } else if (data.type === "error") {
                        textDiv.textContent = "Error: " + (data.content || "Unknown error");
                    }
                }
            }
        } catch (err) {
            textDiv.textContent = "Error: " + err.message;
        } finally {
            isStreaming = false;
            sendBtn.disabled = false;
            chatInput.focus();
        }
    };
})();
```

- [ ] **Step 3: Run UI test to verify content**

```bash
pytest tests/integration/test_ui_routes.py::test_chat_page_returns_200_with_content -v
```

Expected: PASSED ("Konteks Klasifikasi" is in the page).

- [ ] **Step 4: Run full test suite**

```bash
pytest -q --ignore=tests/integration/test_e2e_smoke.py
```

Expected: 67 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add templates/chat.html static/js/chat.js
git commit -m "feat(ui): chat page with SSE streaming, citation list, disclaimer

Consultation layout. Context header from URL params (session/label/
confidence). Chat bubbles (user right teal-tinted, assistant left
bordered). Streaming via fetch + ReadableStream parsing (not EventSource
— POST SSE requires fetch). Citation as text links [N] title — source.
Persistent amber disclaimer above input. Enter to send, Shift+Enter
newline.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Plan Self-Review

**1. Spec coverage:**

| Spec section | Task(s) |
|---|---|
| §2 Architecture (Jinja2 + Tailwind + vanilla JS) | Task 2 (routes), Task 3 (base.html) |
| §3 Design principia (anti-slop, color, typography) | Task 3 (base + CSS), Task 4 (home), Task 5 (upload), Task 6 (chat) |
| §3.5 Dark mode | Task 3 (theme.js, navbar toggle, dark: classes in base) |
| §4.1 Home page | Task 4 |
| §4.2 Upload page | Task 5 |
| §4.3 Chat page | Task 6 |
| §4.4 SSE flow (fetch + ReadableStream) | Task 6 (chat.js) |
| §5 Refactor utils/ → services/ | Task 1 |
| §6 Testing (3 UI tests + refactor verification) | Task 1 (64 pass), Task 2 (3 new tests) |
| §7 Risks | Mitigated: grep check (Task 1 Step 4), fetch for POST SSE (Task 6) |
| §8 Implementation order | Matches task order 1-6 |

**2. Placeholder scan:** No TBDs, TODOs, or "implement later". All code blocks are complete. ✓

**3. Type consistency:**
- `router` (APIRouter) used in Task 2, imported in main.py Task 2 Step 4 ✓
- `templates.TemplateResponse("home.html", {"request": request})` — Jinja2 signature consistent ✓
- Chat query params: `session`, `label`, `confidence` — consistent between ui_routes.py (Task 2), chat.html (Task 6), upload.js (Task 5) ✓
- `from services.rag.app_state import initialize_app_state` — consistent after Task 1 refactor ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-20-ui-and-folder-restructure.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session, batch execution with checkpoints

Which approach?
