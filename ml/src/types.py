from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ScorerType(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    CROSS_ENCODER = "cross_encoder"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ContextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    source: str | None = None


class ChunkTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: Literal["system", "history", "context"]
    source: str | None = None
    original_index: int
    text: str
    token_count: int
    selected: bool
    protected: bool = False
    score: float = 0.0
    reason: str = ""


class CompressionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    compressed_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    saved_tokens: int = Field(ge=0)
    compression_ms: float = Field(ge=0.0)
    budget_exceeded: bool = False
    token_budget: int = Field(ge=1)
    tokenizer: Literal["cl100k_base", "heuristic"]
    selected_chunks: list[ChunkTrace] = Field(default_factory=list)
    dropped_chunks: list[ChunkTrace] = Field(default_factory=list)
