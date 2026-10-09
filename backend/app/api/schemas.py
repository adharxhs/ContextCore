"""HTTP boundary models for the public ContextCore contract."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ContextBlock(BaseModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str | None = None


class CompressRequest(BaseModel):
    system_prompt: str = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)
    context_blocks: list[ContextBlock] = Field(default_factory=list)
    query: str = Field(min_length=1)
    token_budget: int = Field(gt=0)
    scorer: str = "hybrid"

    @field_validator("scorer")
    @classmethod
    def normalize_scorer(cls, value: str) -> str:
        return value.strip().lower()


class ChunkTrace(BaseModel):
    id: str
    source_type: str
    source: str | None = None
    original_index: int
    text: str
    token_count: int = Field(ge=0)
    selected: bool
    protected: bool
    score: float
    reason: str


class CompressionResponse(BaseModel):
    compressed_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    saved_tokens: int
    compression_ms: float = Field(ge=0)
    selected_chunks: list[ChunkTrace]
    dropped_chunks: list[ChunkTrace]
