import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.chat import ChatChunk, ChatRequest
from services.rag.app_state import get_chain, get_memory
from services.rag.citation import format_for_ui
from services.rag.disclaimer import DISCLAIMERS, force_append_disclaimer
from services.rag.language import detect_language
from services.rag.safety import classify_query_danger

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


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(request: ChatRequest) -> StreamingResponse:
    language = detect_language(request.message)
    classification = classify_query_danger(request.message, language)

    if classification in _BLOCKED_RESPONSES:
        blocked_text = _BLOCKED_RESPONSES[classification][language]
        blocked_chunk = ChatChunk(type="blocked", content=blocked_text, language=language)

        async def blocked_stream():
            yield _sse(blocked_chunk.model_dump())
            yield _sse(ChatChunk(type="done").model_dump())

        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    chain = get_chain()
    memory = get_memory()
    chat_history = memory.get_history(request.session_id)
    detection_str = (
        request.detection.model_dump_json() if request.detection else "No prior detection."
    )

    chain_input = {
        "question": request.message,
        "chat_history": chat_history,
        "detection": detection_str,
        "language": language,
        "disclaimer": DISCLAIMERS[language],
    }

    async def event_stream():
        start = time.time()
        full_response = ""
        retrieved_docs: list = []
        try:
            async for event in chain.astream_events(
                chain_input, version="v2"
            ):
                kind = event["event"]
                if kind == "on_chain_end":
                    output = event["data"].get("output")
                    if (
                        isinstance(output, dict)
                        and "retrieved_docs" in output
                        and retrieved_docs == []
                    ):
                        retrieved_docs = output["retrieved_docs"]
                        if retrieved_docs:
                            chunks_meta = [{"metadata": d.metadata} for d in retrieved_docs]
                            citations = format_for_ui(
                                chunks_meta, list(range(1, len(retrieved_docs) + 1))
                            )
                            yield _sse(ChatChunk(
                                type="citation",
                                citations=citations,
                                language=language,
                            ).model_dump())
                elif kind == "on_llm_stream":
                    chunk = event["data"].get("chunk")
                    token = getattr(chunk, "content", "") or ""
                    if token:
                        full_response += token
                        yield _sse(ChatChunk(
                            type="token", content=token, language=language
                        ).model_dump())

            full_response = force_append_disclaimer(full_response, language)
            await memory.add_turn(request.session_id, request.message, full_response)

            from services.rag.logging_config import log_query
            # TODO: tokens_in/tokens_out require non-streaming token counts; not yet wired.
            log_query(
                session_id=request.session_id,
                query=request.message,
                language=language,
                num_chunks_retrieved=len(retrieved_docs),
                num_chunks_after_filter=len(retrieved_docs),
                citations_used=[],
                tokens_in=0,
                tokens_out=0,
                response_time_ms=int((time.time() - start) * 1000),
            )

            yield _sse(ChatChunk(type="done", language=language).model_dump())
        except Exception as e:
            yield _sse(ChatChunk(type="error", content=str(e), language=language).model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")
