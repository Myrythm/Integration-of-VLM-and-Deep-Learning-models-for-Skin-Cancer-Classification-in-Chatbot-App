import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas.chat import ChatChunk, ChatRequest
from utils.rag.app_state import get_chain, get_memory
from utils.rag.citation import extract_citations, format_for_ui
from utils.rag.disclaimer import DISCLAIMERS, force_append_disclaimer
from utils.rag.language import detect_language
from utils.rag.safety import classify_query_danger

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

    full_response = ""
    retrieved_docs: list[dict] = []

    async def event_stream():
        nonlocal full_response
        try:
            async for chunk in chain.astream({
                "question": request.message,
                "chat_history": chat_history,
                "detection": detection_str,
                "language": language,
                "disclaimer": DISCLAIMERS[language],
            }):
                if isinstance(chunk, dict) and "retrieved_docs" in chunk:
                    retrieved_docs.extend(chunk["retrieved_docs"])
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_response += token
                yield _sse(ChatChunk(type="token", content=token, language=language).model_dump())

            full_response = force_append_disclaimer(full_response, language)
            await memory.add_turn(request.session_id, request.message, full_response)

            cited = extract_citations(full_response)
            if cited and retrieved_docs:
                chunks_meta = [{"metadata": d.metadata} for d in retrieved_docs]
                yield _sse(ChatChunk(
                    type="citation",
                    citations=format_for_ui(chunks_meta, cited),
                    language=language,
                ).model_dump())

            yield _sse(ChatChunk(type="done", language=language).model_dump())
        except Exception as e:
            yield _sse(ChatChunk(type="error", content=str(e), language=language).model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")
