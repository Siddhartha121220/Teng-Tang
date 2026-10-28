"""
app.py — FastAPI backend for the Tanglish / Tenglish autocomplete UI.

Endpoints:
  GET  /            → static/index.html  (the autocomplete UI)
  POST /complete    → {"text": "...", "language": "tanglish"|"tenglish"} → {"completion": "..."}
  POST /suggest     → {"text": "...", "language": "tanglish"|"tenglish"} → {"words": [...]}
  GET  /languages   → {"languages": ["tanglish", "tenglish"]}
  GET  /health      → {"status": "ok", "device": "cpu|cuda"}

Run:
    uvicorn app:app --reload
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from inference import MODEL_DIRS, _load_model, generate_completion, generate_suggestions

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

Language = Literal["tanglish", "tenglish"]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app):
    # Load both models up front, so the first suggestion (or language switch) doesn't wait on loading.
    for language in MODEL_DIRS:
        try:
            _load_model(language)
        except RuntimeError as e:
            logger.warning("%s model not loaded: %s", language, e)
    yield


app = FastAPI(
    title="Tanglish / Tenglish Autocomplete",
    description="DistilGPT2 fine-tuned on Tamil-English and Telugu-English code-mixed text",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Static files ─────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class CompleteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="Naan gym ku pogaren")
    language: Language = "tanglish"
    max_new_tokens: int = Field(default=15, ge=1, le=100)


class CompleteResponse(BaseModel):
    completion: str


class SuggestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="Naan gym ku")
    language: Language = "tanglish"


class SuggestResponse(BaseModel):
    words: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_ui():
    index = _static_dir / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="static/index.html not found")
    return FileResponse(str(index))


@app.get("/health")
async def health():
    import inference as _inf
    return {"status": "ok", "device": _inf._device}


@app.get("/languages")
async def languages():
    return {"languages": list(MODEL_DIRS)}


@app.post("/complete", response_model=CompleteResponse)
async def complete(req: CompleteRequest):
    try:
        completion = generate_completion(
            prompt=req.text,
            language=req.language,
            max_new_tokens=req.max_new_tokens,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return CompleteResponse(completion=completion)


@app.post("/suggest", response_model=SuggestResponse)
async def suggest(req: SuggestRequest):
    """Return top 3 most likely next words / word completions."""
    try:
        words = generate_suggestions(prompt=req.text, language=req.language, k=3)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Suggest error")
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {e}")

    return SuggestResponse(words=words)
