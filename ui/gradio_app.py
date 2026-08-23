"""
Gradio chat UI for the retail agent.

Mounted on the FastAPI app at /ui. Users can ask store questions and
optionally pick an LLM provider when more than one is configured.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import gradio as gr


def available_providers() -> List[str]:
    """Providers that can run with the current environment."""
    import os

    options = ["mock"]
    if os.getenv("OPENAI_API_KEY"):
        options.append("openai")
    if os.getenv("DEEPSEEK_API_KEY"):
        options.append("deepseek")
    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
        options.append("bedrock")
    if os.getenv("VERTEX_PROJECT_ID"):
        options.append("vertex")
    return options


def build_gradio_app(
    ask: Callable[[str, str], dict],
    default_provider: str,
) -> gr.Blocks:
    """
    Build the Gradio Blocks app.

    ask(query, provider) -> {"answer", "sources", "tool_info"}
    """
    providers = available_providers()
    if default_provider not in providers:
        providers = [default_provider] + [p for p in providers if p != default_provider]

    can_choose = len(providers) > 1
    provider_label = (
        f"LLM provider (default from .env: {default_provider})"
        if can_choose
        else f"LLM provider (fixed): {default_provider}"
    )

    def respond(message: str, history: list, provider: str):
        if not message or not message.strip():
            yield history, ""
            return

        provider = (provider or default_provider).lower()
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

            extras = []
            if guardrail.get("blocked"):
                extras.append(
                    f"Guardrail blocked ({guardrail.get('stage', '?')}: "
                    f"{guardrail.get('code', 'unknown')})"
                )
            if sources:
                extras.append("Sources: " + ", ".join(str(s) for s in sources))
            if tool_info:
                extras.append(tool_info)
            if extras:
                answer = answer + "\n\n---\n" + "\n".join(extras)
        except Exception as e:
            answer = f"Error: {e}"

        history[-1] = {"role": "assistant", "content": answer}
        yield history, ""

    with gr.Blocks(title="Retail Agent") as demo:
        gr.Markdown(
            "# Retail Agent\n"
            "Ask about products, stock, shipping, returns, and recommendations "
            "for a Pakistani clothing store in Germany."
        )

        provider = gr.Dropdown(
            choices=providers,
            value=default_provider if default_provider in providers else providers[0],
            label=provider_label,
            interactive=can_choose,
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
