from __future__ import annotations

import os
from pathlib import Path


PRODUCT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(
    os.getenv(
        "DATA_PATH",
        PRODUCT_ROOT / "data" / "Datos_consolidados_siniestralidad_vial.xlsx",
    )
)
LOCAL_METADATA_PATH = PRODUCT_ROOT / "artifacts" / "model_metadata.json"
API_URL = os.getenv("MODEL_API_URL", "http://localhost:8000").rstrip("/")

