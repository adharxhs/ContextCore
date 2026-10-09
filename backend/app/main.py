from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backend.app import config
from backend.app.api.schemas import CompressionResponse, CompressRequest, ErrorResponse
from backend.app.services.compression import CompressionServiceError, compress

app = FastAPI(title="ContextCore", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error(code: str, message: str, details: list[dict] | None = None) -> dict:
    return ErrorResponse(error={"code": code, "message": message, "details": details}).model_dump(
        exclude_none=True
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error("invalid_request", "Request validation failed.", exc.errors()),
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/compress",
    response_model=CompressionResponse,
    response_model_exclude_none=True,
    tags=["compression"],
)
def compress_context(request: CompressRequest, response: Response) -> dict:
    """Return selected evidence and a full provenance trace without an LLM call."""
    try:
        result, execution_mode = compress(request)
        result["execution_mode"] = execution_mode
        validated = CompressionResponse.model_validate(result)
        response.headers["X-ContextCore-Execution"] = execution_mode
        if execution_mode == "fallback":
            response.headers["X-ContextCore-Tokenizer"] = "heuristic"
        return validated.model_dump(exclude_none=True)
    except CompressionServiceError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(exc.code, exc.message),
            headers={"X-ContextCore-Execution": "engine"},
        )
    except ValidationError:
        return JSONResponse(
            status_code=502,
            content=_error(
                "invalid_engine_response",
                "The engine returned a response outside the API contract.",
            ),
            headers={"X-ContextCore-Execution": "engine"},
        )
    except Exception:
        return JSONResponse(
            status_code=500,
            content=_error(
                "compression_service_failed",
                "The compression service could not complete the request.",
            ),
        )
