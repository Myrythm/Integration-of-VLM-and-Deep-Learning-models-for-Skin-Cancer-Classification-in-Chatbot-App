import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from config import get_settings
from schemas.chat import ChatChunk, ChatRequest
from services.rag.app_state import get_chain, get_memory
from services.rag.citation import extract_citations, format_for_ui
from services.rag.disclaimer import DISCLAIMERS, force_append_disclaimer
from services.rag.language import detect_language
from services.rag.logging_config import log_query
from services.rag.safety import classify_query_danger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


_BLOCKED_RESPONSES = {
    "unsafe_dosage": {
        "en": "I cannot provide medication dosage advice. Please consult a pharmacist or doctor.",
        "id": "Saya tidak bisa memberikan saran dosis obat. Silakan konsultasi dengan apoteker atau dokter.",
    },
    "off_topic": {
        "en": "I can only help with questions about skin health that was screened. For other topics, please consult a relevant professional.",
        "id": "Saya hanya bisa membantu pertanyaan terkait kesehatan kulit yang terdeteksi. Untuk topik lain, silakan konsultasi profesional terkait.",
    },
}

_ERROR_RESPONSES = {
    "en": "Sorry, something went wrong. Please try again.",
    "id": "Maaf, terjadi kesalahan. Silakan coba lagi.",
}


# Defeat proxy buffering (nginx et al.) that would otherwise batch the token stream.
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _sse(chunk: ChatChunk) -> str:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def _render_detection(detection) -> str:
    """Render the detection into the system prompt safely.

    The detection comes from the client; never serialize the whole object into the
    highest-privilege prompt slot (prompt-injection channel). Only the validated,
    constrained fields are rendered.
    """
    if detection is None:
        return "No prior detection."
    return f"label={detection.label}, confidence={detection.confidence:.4f}"


@router.post("")
async def chat(request: ChatRequest) -> StreamingResponse:
    language = detect_language(request.message)
    classification = classify_query_danger(request.message, language)

    if classification in _BLOCKED_RESPONSES:
        return StreamingResponse(
            _blocked_stream(request, classification, language),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    chain = get_chain()
    memory = get_memory()
    chat_history = await memory.get_history(request.session_id)

    chain_input = {
        "question": request.message,
        "chat_history": chat_history,
        "detection": _render_detection(request.detection),
        "language": language,
        "disclaimer": DISCLAIMERS[language],
    }
    timeout_s = get_settings().chat_stream_timeout_seconds

    return StreamingResponse(
        _answer_stream(request, chain, memory, chain_input, language, timeout_s),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _blocked_stream(request: ChatRequest, classification: str, language: str):
    start = time.time()
    blocked_text = force_append_disclaimer(
        _BLOCKED_RESPONSES[classification][language], language
    )
    memory = get_memory()
    try:
        yield _sse(ChatChunk(type="blocked", content=blocked_text, language=language))
        await memory.add_turn(request.session_id, request.message, blocked_text)
        log_query(
            session_id=request.session_id,
            query=request.message,
            language=language,
            num_chunks_retrieved=0,
            num_chunks_after_filter=0,
            citations_used=[],
            tokens_in=0,
            tokens_out=0,
            response_time_ms=int((time.time() - start) * 1000),
            classification=classification,
        )
    finally:
        try:
            yield _sse(ChatChunk(type="done", language=language))
        except Exception:
            pass


async def _answer_stream(request, chain, memory, chain_input, language, timeout_s):
    start = time.time()
    parts: list[str] = []
    retrieved_docs: list = []
    seen_retrieve = False
    tokens_in = 0
    tokens_out = 0
    try:
        try:
            async with asyncio.timeout(timeout_s):
                async for event in chain.astream_events(chain_input, version="v2"):
                    kind = event["event"]
                    if (
                        kind == "on_chain_end"
                        and event.get("name") == "retrieve"
                        and not seen_retrieve
                    ):
                        output = event["data"].get("output")
                        if isinstance(output, dict) and "retrieved_docs" in output:
                            retrieved_docs = output["retrieved_docs"]
                            seen_retrieve = True
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        usage = getattr(chunk, "usage_metadata", None)
                        if usage:
                            tokens_in = usage.get("input_tokens", tokens_in)
                            tokens_out = usage.get("output_tokens", tokens_out)
                        token = getattr(chunk, "content", "") or ""
                        if token:
                            parts.append(token)
                            yield _sse(
                                ChatChunk(type="token", content=token, language=language)
                            )
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning("chat stream timed out after %ss", timeout_s)
            yield _sse(
                ChatChunk(type="error", content=_ERROR_RESPONSES[language], language=language)
            )
            return

        full_response = force_append_disclaimer("".join(parts), language)
        cited = extract_citations(full_response)

        if retrieved_docs and cited:
            chunks_meta = [{"metadata": d.metadata} for d in retrieved_docs]
            citations = format_for_ui(chunks_meta, cited)
            if citations:
                yield _sse(
                    ChatChunk(type="citation", citations=citations, language=language)
                )

        await memory.add_turn(request.session_id, request.message, full_response)

        prefilter_count = (
            retrieved_docs[0].metadata.get("prefilter_count", len(retrieved_docs))
            if retrieved_docs
            else 0
        )
        log_query(
            session_id=request.session_id,
            query=request.message,
            language=language,
            num_chunks_retrieved=prefilter_count,
            num_chunks_after_filter=len(retrieved_docs),
            citations_used=cited,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            response_time_ms=int((time.time() - start) * 1000),
            classification="answered",
        )
    except asyncio.CancelledError:
        logger.info("chat stream cancelled by client disconnect")
        raise
    except Exception:
        logger.exception("chat stream failed")
        yield _sse(
            ChatChunk(type="error", content=_ERROR_RESPONSES[language], language=language)
        )
    finally:
        try:
            yield _sse(ChatChunk(type="done", language=language))
        except Exception:
            pass
