"""Pydantic schemas for document endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

class DocumentCreateResponse(BaseModel):
    id: UUID = Field(..., description="Document UUID")
    filename: str
    uploaded_at: datetime

class DocumentInfo(BaseModel):
    id: UUID
    filename: str
    content_type: Optional[str] = None
    uploaded_at: datetime
    # In a real system we would not return raw content here.

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

