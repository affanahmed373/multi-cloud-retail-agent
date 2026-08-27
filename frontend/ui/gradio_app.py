"""
Gradio chat UI for the retail agent.

Mounted on the FastAPI app at /ui. Users can pick OpenAI, DeepSeek, Gemini,
Vertex AI, Bedrock, or Mock for each question.
"""

from __future__ import annotations

from typing import Callable, List

import gradio as gr

from backend.app.provider_registry import (
    default_provider_choice,
    normalize_provider,
    provider_choices,
    provider_help_text,
)


def build_gradio_app(
    ask: Callable[[str, str], dict],
    default_provider: str | None = None,
) -> gr.Blocks:
    """
    Build the Gradio Blocks app.

    ask(query, provider) -> {"answer", "sources", "tool_info", "guardrail"}
    """
    choices = provider_choices(include_mock=True)
    provider_ids = [pid for _, pid in choices]
    default = default_provider_choice(default_provider or "mock")
    if default not in provider_ids:
        default = provider_ids[0]

    def respond(message: str, history: list, provider: str):
        if not message or not message.strip():
            yield history, ""
            return

        provider = normalize_provider(provider or default)
        history = list(history or [])
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Thinking…"})
        yield history, ""

        try:
            result = ask(message.strip(), provider)
            answer = result.get("answer") or "(No answer)"
            sources = result.get("sources") or []
            tool_info = (result.get("tool_info") or "").strip()
            guardrail = result.get("guardrail") or {}

            extras = [f"Provider: {provider}"]
            if guardrail.get("blocked"):
                extras.append(
                    f"Guardrail blocked ({guardrail.get('stage', '?')}: "
                    f"{guardrail.get('code', 'unknown')})"
                )
            elif guardrail.get("pii_redacted_input") or guardrail.get(
                "pii_redacted_output"
            ):
                pii = guardrail.get("pii_detected") or []
                if pii:
                    extras.append("PII redacted: " + ", ".join(pii))
            if sources:
                extras.append("Sources: " + ", ".join(str(s) for s in sources))
            if tool_info:
                extras.append(tool_info)
            if extras:
                answer = answer + "\n\n---\n" + "\n".join(extras)
        except Exception as e:
            answer = f"Error ({provider}): {e}"

        history[-1] = {"role": "assistant", "content": answer}
        yield history, ""

    with gr.Blocks(title="Retail Agent") as demo:
        gr.Markdown(
            "# Retail Agent\n"
            "Ask about products, stock, shipping, returns, and recommendations "
            "for a Pakistani clothing store in Germany."
        )
        gr.Markdown(provider_help_text())

        provider = gr.Dropdown(
            choices=choices,
            value=default,
            label="LLM provider",
            interactive=True,
        )

        chatbot = gr.Chatbot(label="Chat", height=480)
        msg = gr.Textbox(
            label="Your question",
            placeholder="e.g. Do you ship to Berlin?",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Ask", variant="primary")
            clear = gr.Button("Clear")

        gr.Examples(
            examples=[
                "Do you ship to Berlin?",
                "What is your return policy?",
                "Do you have black shalwar kameez in size M?",
                "I need something for a summer wedding under 80 EUR.",
            ],
            inputs=msg,
        )

        send.click(respond, [msg, chatbot, provider], [chatbot, msg])
        msg.submit(respond, [msg, chatbot, provider], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo
