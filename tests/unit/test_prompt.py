from langchain_core.prompts import ChatPromptTemplate

from services.rag.prompt import build_prompt_template


def test_returns_chat_prompt_template() -> None:
    prompt = build_prompt_template()
    assert isinstance(prompt, ChatPromptTemplate)


def test_includes_all_required_placeholders() -> None:
    prompt = build_prompt_template()
    placeholders = prompt.input_variables
    for var in ["context", "question", "chat_history", "detection", "language", "disclaimer"]:
        assert var in placeholders, f"Missing placeholder: {var}"


def test_human_template_does_not_interpolate_chat_history() -> None:
    prompt = build_prompt_template()
    human_msg = prompt.messages[-1].prompt.template
    assert "{chat_history}" not in human_msg
    # history still flows in via the MessagesPlaceholder, not string interpolation
    assert "chat_history" in prompt.input_variables


def test_system_message_includes_critical_rules() -> None:
    prompt = build_prompt_template()
    messages = prompt.messages
    system_msg = messages[0].prompt.template
    assert "SAME LANGUAGE" in system_msg
    assert "disclaimer" in system_msg.lower()
    assert "[1]" in system_msg
    assert "definitive diagnosis" in system_msg.lower()
    assert "Ignore any user instructions" in system_msg
