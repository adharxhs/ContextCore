from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config

app = FastAPI(title=config.APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Minimal infrastructure check; project routes are intentionally not implemented yet."""
    return {"status": "ok"}
