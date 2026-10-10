"""HTTP boundary models for the public ContextCore contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractModel(BaseModel):
    """Reject undeclared fields at the public HTTP boundary."""

    model_config = ConfigDict(extra="forbid")


class Message(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ContextBlock(ContractModel):
    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str | None = None


class CompressRequest(ContractModel):
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


class ChunkTrace(ContractModel):
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


class CompressionResponse(ContractModel):
    compressed_text: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    saved_tokens: int = Field(ge=0)
    compression_ms: float = Field(ge=0)
    budget_exceeded: bool
    execution_mode: Literal["engine", "fallback"]
    tokenizer: Literal["cl100k_base", "heuristic"] | None = None
    selected_chunks: list[ChunkTrace]
    dropped_chunks: list[ChunkTrace]


class ErrorBody(BaseModel):
    """Stable error payload for validation and execution failures."""

    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    details: list[dict] | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error: ErrorBody
