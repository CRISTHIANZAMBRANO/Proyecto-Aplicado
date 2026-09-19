from __future__ import annotations

import os
from pathlib import Path


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = PRODUCT_ROOT / "artifacts"

MODEL_ARTIFACT_PATH = Path(
    os.getenv("MODEL_ARTIFACT_PATH", DEFAULT_ARTIFACT_DIR / "model_pipeline.joblib")
)
MODEL_METADATA_PATH = Path(
    os.getenv("MODEL_METADATA_PATH", DEFAULT_ARTIFACT_DIR / "model_metadata.json")
)

API_TITLE = "API de Siniestralidad Vial"
API_VERSION = "1.1.0"


def allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501")
    return [value.strip() for value in raw.split(",") if value.strip()]
