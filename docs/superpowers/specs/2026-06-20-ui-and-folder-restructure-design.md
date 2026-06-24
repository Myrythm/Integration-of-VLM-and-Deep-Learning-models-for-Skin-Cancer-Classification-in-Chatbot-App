# UI + Folder Restructure Design Spec

**Date:** 2026-06-20
**Status:** Approved
**Author:** Brainstorming session
**Target Project:** `Myrythm/Integration-of-GPT-4o-Vision-and-EfficientNetB3-for-Skin-Cancer-Classification-in-Chatbot-App`

---

## 1. Overview

### 1.1 Background

v2 RAG-enhanced skin cancer chatbot is functional via API (`POST /api/upload`, `POST /api/chat`) but has no UI. The `utils/` folder name is generic; user wants `services/` for descriptive business-logic grouping. This spec adds a 3-page UI (Home, Upload, Chat) with dark mode support and restructures `utils/` → `services/`.

### 1.2 Goals

1. **3-page MVP UI** — Home (landing), Upload (image classification), Chat (RAG chatbot with SSE streaming, citation, disclaimer).
2. **Clinical design, not AI slop** — medical/clinical aesthetic specific to skin cancer screening for Indonesian patients.
3. **Dark mode** — toggle in navbar, persists via localStorage, respects OS preference on first visit.
4. **Restructure `utils/` → `services/`** — descriptive folder name, `services/image/` and `services/rag/` subfolders.
5. **Tailwind via CDN** — no build step, no npm.
6. **Indonesian UI chrome** — navbar, buttons, labels in Bahasa Indonesia. Chatbot response stays bilingual.
7. **Preserve all existing tests** — 64/64 must still pass after refactor.

### 1.3 Non-Goals (YAGNI)

- No article pages, no FAQ section.
- No bilingual UI toggle (UI chrome is Indonesian only; dark mode toggle only).
- No browser automation tests (Selenium/Playwright).
- No Tailwind build step — CDN only.
- No Alpine.js / HTMX — vanilla JS only.

---

## 2. Architecture

### 2.1 Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Template engine | Jinja2 (via starlette) | Already in requirements.txt, FastAPI native |
| CSS | Tailwind CDN (`cdn.tailwindcss.com`) | No build, no npm, class-based, `darkMode: 'class'` |
| JS | Vanilla JS (EventSource/fetch + theme toggle) | ~120 lines total across 3 files |
| Font | Inter (Google Fonts CDN) | Clean, readable, document-like |
| Color palette | Medical teal + warm neutral + amber + dark slate | Trustworthy, clinical, WCAG AA compliant |

### 2.2 Folder Structure (post-refactor + new UI)

```
project-root/
├── main.py                          # FastAPI app (mount static, register routers, lifespan)
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
│
├── services/                        # NEW (replaces utils/) — business logic
│   ├── __init__.py
│   ├── image/
│   │   ├── __init__.py
│   │   └── classifier.py            # MOVED from utils/image_classifier.py
│   └── rag/                         # MOVED from utils/rag/
│       ├── __init__.py
│       ├── app_state.py
│       ├── cache.py
│       ├── chain.py
│       ├── citation.py
│       ├── disclaimer.py
│       ├── embedder.py
│       ├── ingestion.py
│       ├── language.py
│       ├── llm_provider.py
│       ├── logging_config.py
│       ├── memory.py
│       ├── prompt.py
│       ├── pubmed.py
│       ├── retriever.py
│       ├── safety.py
│       └── vector_store.py
│
├── routes/
│   ├── __init__.py
│   ├── api_routes.py                # POST /api/upload (update imports)
│   ├── chat_routes.py               # POST /api/chat (update imports)
│   └── ui_routes.py                 # NEW — GET /, GET /upload, GET /chat
│
├── schemas/                         # unchanged
│   ├── chat.py
│   ├── detection.py
│   └── image.py
│
├── templates/                       # NEW — Jinja2 HTML
│   ├── base.html                    # Shared layout (head, Tailwind CDN+config, navbar, footer)
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
│       ├── upload.js
│       ├── chat.js
│       └── theme.js                 # NEW — dark mode toggle
│
├── data/                            # unchanged
├── tests/                           # update imports (utils.* → services.*)
├── docs/                            # unchanged
├── scripts/                         # unchanged
├── model/                           # unchanged
└── logs/                            # gitignored
```

### 2.3 Module Boundaries

- **`services/`** = framework-agnostic business logic (no FastAPI imports).
- **`routes/`** = thin HTTP layer. `ui_routes.py` renders Jinja2; `api_routes.py` and `chat_routes.py` return JSON/SSE.
- **`schemas/`** = Pydantic data shapes only.
- **`templates/`** = Jinja2 HTML, rendered by `routes/ui_routes.py`.
- **`static/`** = CSS/JS served at `/static/*`.

### 2.4 main.py changes

Re-add `StaticFiles` mount and `Jinja2Templates`, register `ui_router`:

```python
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes.ui_routes import router as ui_router

templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Skin Cancer RAG Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(ui_router)
app.include_router(api_router)
app.include_router(chat_router)
```

---

## 3. Design Principia (Anti-AI-Slop)

### 3.1 What we avoid

| AI slop pattern | Why avoid |
|---|---|
| Gradient hero | Trendy, characterless |
| "AI-powered" copy | Selling point is screening, not tech |
| 3-column feature grid with icons | Generic SaaS template |
| Generic "Get Started" CTA | Non-specific |
| Empty hero with tagline only | No information value |
| Glassmorphism / trendy effects | Date fast |
| Lorem ipsum / placeholder copy | Says nothing |
| Generic chat bubbles with avatars | Not consultation feel |

### 3.2 What we use instead

| Clinical design choice | Rationale |
|---|---|
| Solid warm neutral + medical teal | Trustworthy, clinical, not trendy |
| No "AI" mention in copy | Focus on screening value |
| Linear info flow, definition lists | Information-dense, document-like |
| Specific CTAs: "Upload Foto Lesi", "Klasifikasikan" | Action-oriented, domain-specific |
| Hero with real info (jenis kanker) | Content carries the page |
| Flat, document-like layout | Medical report aesthetic |
| Bahasa Indonesia: Sp.KK, faktor risiko | Indonesian healthcare context |
| Consultation layout for chat | Patient summary header, references list |

### 3.3 Color Palette — Light + Dark

```
                     LIGHT MODE              DARK MODE
--bg:               #fafaf9 (warm white)    #0f172a (slate-900, not pure black)
--surface:          #ffffff                 #1e293b (slate-800, elevated from bg)
--text:             #1e293b (dark slate)    #f1f5f9 (slate-100, warm white)
--text-muted:       #64748b                 #94a3b8 (slate-400)
--primary:          #0d9488 (teal-600)      #14b8a6 (teal-400, brighter for contrast)
--primary-dark:     #0f766e (teal-700)      #0d9488 (teal-600, hover in dark)
--accent:           #d97706 (amber-600)     #f59e0b (amber-500, brighter for visibility)
--success:          #15803d (green-700)     #22c55e (green-500, brighter)
--border:           #e2e8f0 (slate-200)     #334155 (slate-700, subtle in dark)
```

No gradients. Solid colors. Dark mode uses slate-900 (not pure black) to maintain clinical softness. Teal brightens in dark for WCAG AA contrast.

### 3.4 Typography

- **Body**: Inter (Google Fonts CDN), weights 400/500/600
- **Headings**: Inter 700, `tracking-tight`
- **Monospace**: `ui-monospace, SFMono-Regular` (system, no CDN)
- **Size scale**: `text-sm` (14px) default, `text-base` (16px) important, `text-2xl` max for titles. No `text-6xl`.

### 3.5 Dark Mode Implementation

**Strategy**: Tailwind `darkMode: 'class'` + `<html class="dark">` toggle.

```html
<!-- base.html: configure before CDN script -->
<script>tailwind = { config: { darkMode: 'class' } }</script>
<script src="https://cdn.tailwindcss.com"></script>
```

**Toggle UX**:
- Sun/moon icon (inline SVG, 16x16, no icon library) in navbar right
- `localStorage.setItem('theme', 'dark'|'light')` persists
- First visit: check `prefers-color-scheme: dark` (OS preference), then save
- `transition-colors duration-200` on body for smooth swap
- `static/js/theme.js` (~20 lines): apply on load, toggle on click, swap icon

**Files affected**:
| File | Change |
|------|--------|
| `templates/base.html` | Tailwind config `darkMode: 'class'`, init script for `<html>` class |
| `templates/partials/navbar.html` | Toggle button + sun/moon SVG |
| `static/js/theme.js` (NEW) | Toggle logic + localStorage + OS detection |
| `static/css/app.css` | `transition-colors` on body |
| All templates | `dark:` variants on bg, text, border elements |

---

## 4. Page Designs

### 4.1 Home (`/`)

Document-style, single column, information-dense.

**Sections**:
1. **Title**: "Skrining Kanker Kulit berbasis EfficientNetB3" (`text-2xl`, `font-bold`, `tracking-tight`)
2. **Subtitle**: "Upload foto lesi kulit untuk klasifikasi 4 jenis: BCC, SCC, Melanoma, Nevus. Hasil dilanjutkan ke chatbot edukasi bilingual." (`text-sm`, `text-muted`)
3. **CTA**: "Upload Foto Lesi" (solid teal button → `/upload`)
4. **"Jenis yang Dideteksi"** — definition list:
   - Karsinoma Sel Basal — Jenis paling umum, tumbuh lambat
   - Karsinoma Sel Skuamosa — Bisa menyebar ke lapisan kulit lebih dalam
   - Melanoma — Paling serius, butuh deteksi dini
   - Nevus — Tahi lalat jinak, umumnya tidak berbahaya
5. **"Kapan Harus ke Dokter"** — bullet list:
   - Lesi berubah bentuk, warna, atau ukuran
   - Berdarah atau gatal tanpa sebab
   - Tahi lalat baru setelah usia 30
   - Lesi yang tidak sembuh dalam 2 minggu
   - Konsultasi ke dokter spesialis kulit (Sp.KK) di rumah sakit terdekat.
6. **Disclaimer bar** — amber: "Alat ini untuk edukasi dan skrining awal, bukan diagnosis medis. Hasil AI bisa keliru."

### 4.2 Upload (`/upload`)

Medical intake form layout.

**Sections**:
1. **Title**: "Upload Foto Lesi"
2. **Drop zone** — dashed teal border, `rounded-lg`:
   - "Pilih atau drop foto di sini"
   - "Format: PNG, JPG, WEBP / Maksimal: 10 MB"
   - "[ Pilih File ]" button inside
3. **"Tips Foto yang Baik"**:
   - Pencahayaan terang, tidak silau
   - Fokus pada lesi, background netral
   - Jarak 10-15 cm, foto tajam
   - Satu lesi per foto
4. **Preview** — thumbnail after file selected
5. **Submit**: "Klasifikasikan" (solid teal, disabled until file)
6. **Result** (after POST /api/upload) — table-style:
   - Diagnosis | Confidence with bar + label (Tinggi ≥0.8 / Sedang 0.5-0.79 / Rendah <0.5)
   - "Model: EfficientNetB3-v1" (muted, monospace)
   - Inline amber: "Hasil AI bisa keliru. Konfirmasi dengan dokter spesialis kulit."
   - "Konsultasi Chatbot" link → `/chat?session={id}&label={label}&confidence={confidence}`

**Loading**: button → "Memproses..." + spinner. No full-screen overlay.

### 4.3 Chat (`/chat`)

Consultation layout.

**Sections**:
1. **Back link**: "← Kembali ke Upload"
2. **Context header**: "Konteks Klasifikasi" — compact card "Melanoma · 96.76% · EfficientNetB3-v1" (monospace). Reads from URL query params.
3. **Chat area** — scrollable bubbles:
   - User: right-aligned, teal-tinted bg, label "User"
   - Assistant: left-aligned, surface bg + border, label "Asisten"
   - Streaming: tokens append real-time, cursor `▌`
   - Citation: "Referensi:" list — `[1] title — source` text links (not cards). Click → `window.open(url, "_blank")`
4. **Input bar**: textarea + `[→]` button. Enter to send, Shift+Enter newline. Disabled while streaming.
5. **Disclaimer bar**: amber, persistent above input: "⚠ Asisten AI untuk edukasi. Bukan nasihat medis. Konsultasi dokter spesialis kulit (Sp.KK)."

### 4.4 SSE Flow (chat.js)

```
1. User types → Enter or click [→]
2. Input disabled, user bubble appears
3. Assistant bubble appears empty with cursor ▌
4. fetch() POST /api/chat with {session_id, message, detection}:
   - Parse ReadableStream for SSE events
   - "citation" → render "Referensi:" list
   - "token" → append text
   - "blocked" → show blocked message
   - "done" → remove cursor, enable input
   - "error" → show error, enable input
5. Citation click → window.open(url, "_blank")
```

**Note**: Standard `EventSource` only supports GET. Use `fetch()` + `ReadableStream` parsing for POST SSE.

---

## 5. Refactor Plan: `utils/` → `services/`

### 5.1 Move operations

| From | To | Method |
|------|-----|--------|
| `utils/image_classifier.py` | `services/image/classifier.py` | `git mv` |
| `utils/rag/*.py` (17 files) | `services/rag/*.py` | `git mv` each |
| `utils/__init__.py` | `services/__init__.py` | `git mv` |
| `utils/rag/__init__.py` | `services/rag/__init__.py` | `git mv` |

Pre-move: remove `__pycache__`. Post-move: delete empty `utils/`.

### 5.2 Import updates

```python
# OLD                                    # NEW
from utils.image_classifier import    → from services.image.classifier import
from utils.rag.                       → from services.rag.
```

~30 files affected (routes, main, tests, eval scripts, internal rag imports).

Verification: `grep -r "from utils" .` returns 0 results.

### 5.3 New: `services/image/__init__.py`

Empty package marker.

---

## 6. Testing Strategy

### 6.1 Refactor verification

Run `pytest` after refactor. 64 existing tests must pass.

### 6.2 New UI tests (3 integration)

| Test | Verifies |
|------|----------|
| `test_home_page` | `GET /` → 200, contains "Skrining Kanker Kulit" |
| `test_upload_page` | `GET /upload` → 200, contains "Upload Foto Lesi" |
| `test_chat_page` | `GET /chat` → 200, contains "Konteks Klasifikasi" |

### 6.3 No JS unit tests

Vanilla JS, ~120 lines, tested manually.

### 6.4 Manual browser test checklist

1. Home renders, CTA → upload
2. Upload: select image → preview → classify → result card
3. Chat: context header → type message → stream → citations clickable
4. Dark mode toggle works, persists on reload
5. Disclaimer visible at all times

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Missed import update | `grep -r "from utils" .` must be 0 |
| `git mv` fails (pycache) | Remove pycache before mv |
| EventSource POST limitation | Use `fetch()` + `ReadableStream` |
| Tailwind CDN slow (~3MB) | Acceptable for demo |
| Dark mode WCAG contrast | Teal/amber brightened in dark palette |
| Jinja2 template errors | UI tests catch 200 status |

---

## 8. Implementation Order

1. Refactor `utils/` → `services/` (git mv + import update + verify 64 tests)
2. Create `routes/ui_routes.py` + register in main.py + re-add StaticFiles/Jinja2Templates
3. Create `templates/` (base with dark mode config, home, upload, chat, partials)
4. Create `static/` (app.css, theme.js, upload.js, chat.js)
5. Create `tests/integration/test_ui_routes.py` (3 tests)
6. Run all tests — 64 + 3 = 67 pass
7. Manual browser test — home → upload → chat + dark mode toggle

---

*End of design spec. Next step: invoke `writing-plans` skill.*
