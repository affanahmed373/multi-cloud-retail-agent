"""
LLM provider interfaces and implementations.

This module defines:
- LLMProvider: abstract interface for generating text.
- MockLLMProvider: rule-based mock for local testing.
- BedrockLLMProvider: placeholder for AWS Bedrock (to implement later).
- VertexLLMProvider: placeholder for GCP Vertex AI (to implement later).
"""

from typing import Dict, Any, Optional


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


class BedrockLLMProvider(LLMProvider):
    """
    Placeholder for AWS Bedrock LLM provider.

    To implement later:
    - Use boto3 to call Bedrock (e.g., Claude Haiku).
    - Pass prompt and system_prompt.
    - Parse response and return text.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
        region_name: str = "us-east-1",
    ) -> None:
        self.model_id = model_id
        self.region_name = region_name
        # TODO: initialize boto3 client here

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        # TODO: implement Bedrock call
        raise NotImplementedError(
            "BedrockLLMProvider is not yet implemented. "
            "This is a placeholder for cloud integration."
        )


class VertexLLMProvider(LLMProvider):
    """
    Placeholder for GCP Vertex AI LLM provider.

    To implement later:
    - Use google-cloud-aiplatform to call Gemini.
    - Pass prompt and system instructions.
    - Parse response and return text.
    """

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ) -> None:
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        # TODO: initialize Vertex AI client here

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        # TODO: implement Vertex AI call
        raise NotImplementedError(
            "VertexLLMProvider is not yet implemented. "
            "This is a placeholder for cloud integration."
        )