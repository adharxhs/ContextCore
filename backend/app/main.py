from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config
from backend.app.api.schemas import CompressRequest, CompressionResponse
from backend.app.services.compression import compress

app = FastAPI(title="ContextCore", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/compress", response_model=CompressionResponse, tags=["compression"])
def compress_context(request: CompressRequest, response: Response) -> dict:
    """Return selected evidence and a full provenance trace without an LLM call."""
    try:
        result, execution_mode = compress(request)
        response.headers["X-ContextCore-Execution"] = execution_mode
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
