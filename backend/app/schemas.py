"""
Pydantic schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatRequest(BaseModel):
    """Request schema for the /chat endpoint."""

    query: str = Field(..., description="User's question or message")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")


class ChatResponse(BaseModel):
    """Response schema for the /chat endpoint."""

    answer: str = Field(..., description="Agent's answer")
    sources: List[str] = Field(default_factory=list, description="List of source IDs used")
    debug: Optional[Dict[str, Any]] = Field(None, description="Optional debug info")