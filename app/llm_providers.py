"""
LLM provider interfaces and implementations.

This module defines:
- LLMProvider: abstract interface for generating text.
- MockLLMProvider: rule-based mock for local testing.
- BedrockLLMProvider: placeholder for AWS Bedrock (to implement later).
- VertexLLMProvider: placeholder for GCP Vertex AI (to implement later).
"""

from typing import Dict, Any, Optional
import boto3
import os
import requests


class LLMProvider:
    """Abstract interface for LLM providers."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for local testing.

    Uses simple rule-based responses to simulate an agent.
    This is enough to test the pipeline without calling real models.
    """

    def __init__(self) -> None:
        pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        q = prompt.lower()

        # Shipping-related
        if "ship" in q and ("berlin" in q or "germany" in q):
            return (
                "Yes, we ship all over Germany including Berlin. "
                "Standard delivery takes 3–5 business days. "
                "Express shipping (1–2 days) is available at checkout for an extra fee."
            )
        if "ship" in q and ("international" in q or "outside" in q):
            return (
                "Yes, we ship to many EU countries and selected international destinations. "
                "Shipping costs and delivery times vary by country and are shown at checkout."
            )

        # Returns
        if "return" in q or "exchange" in q:
            return (
                "You can return items within 14 days of delivery as long as they are "
                "in original, unworn condition with tags attached. We offer refunds or exchanges."
            )

        # Size guide
        if "size" in q and ("guide" in q or "measure" in q):
            return (
                "We provide a detailed size guide with measurements for each product. "
                "You can find it on each product page under 'Size Guide'. "
                "We recommend measuring yourself and comparing with the chart."
            )

        # Payment
        if "payment" in q or "pay" in q:
            return (
                "We accept major credit cards (Visa, Mastercard), PayPal, and bank transfer. "
                "All payments are processed securely."
            )

        # Stock / availability (generic)
        if "stock" in q or "available" in q:
            return (
                "Let me check our inventory. We have several items in stock, "
                "but availability depends on size and color. "
                "Please specify which item and size you're interested in."
            )

        # Recommendations (generic)
        if "recommend" in q or "suggest" in q or "need" in q:
            return (
                "Based on your request, I'd recommend checking our lawn suits, kurtas, "
                "or formal waistcoat sets depending on the occasion and your budget. "
                "Feel free to ask about specific items or price ranges."
            )

        # Fallback
        return (
            "Thank you for your question. Our team is happy to help. "
            "Could you please provide a bit more detail so I can assist you better?"
        )


class OpenAILLMProvider(LLMProvider):
    """
    OpenAI LLM provider using the Chat Completions API.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class GeminiLLMProvider(LLMProvider):
    """
    Google Gemini via the Generative Language API (Google AI Studio).
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": kwargs.get("temperature", 0.3)},
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        response = requests.post(
            url,
            params={"key": self.api_key},
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if part.get("text"))
        if not text:
            raise ValueError("Gemini returned an empty response.")
        return text


class DeepSeekLLMProvider(LLMProvider):
    """
    DeepSeek LLM provider using their OpenAI-compatible Chat Completions API.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set.")
        self.endpoint = "https://api.deepseek.com/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.3),
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
class BedrockLLMProvider(LLMProvider):
    """
    AWS Bedrock LLM provider using the Converse API.
    """

    def __init__(
        self,
        model_id: str = "amazon.nova-lite-v1:0",
        region_name: str = "eu-central-1",
        guardrail_id: Optional[str] = None,
        guardrail_version: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        params: dict = {
            "modelId": self.model_id,
            "messages": messages,
        }

        # Optional guardrail
        if self.guardrail_id and self.guardrail_version:
            params["guardrailConfig"] = {
                "guardrailIdentifier": self.guardrail_id,
                "guardrailVersion": self.guardrail_version,
            }

        response = self.client.converse(**params)

        # Simple text extraction from the output
        content = response["output"]["message"]["content"]
        # Bedrock returns a list of content blocks; assume first is text
        return content[0]["text"]


class VertexLLMProvider(LLMProvider):
    """
    GCP Vertex AI Gemini provider.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        project_id: Optional[str] = None,
        location: str = "europe-west3",
    ) -> None:
        self.model_name = model_name
        self.project_id = (
            project_id
            or os.getenv("VERTEX_PROJECT_ID")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        self.location = location
        if not self.project_id:
            raise ValueError("VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT is not set.")

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=self.project_id, location=self.location)
        self._model = GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self._model.generate_content(
            full_prompt,
            generation_config={"temperature": kwargs.get("temperature", 0.3)},
        )
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Vertex AI returned an empty response.")
        return text

