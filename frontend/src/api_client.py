from __future__ import annotations

import json

import requests

from src.config import API_URL, LOCAL_METADATA_PATH


class ApiUnavailable(RuntimeError):
    pass


def health(timeout: float = 2.0) -> dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ApiUnavailable(str(exc)) from exc


def model_options(timeout: float = 4.0) -> dict:
    try:
        response = requests.get(f"{API_URL}/model/options", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        if not LOCAL_METADATA_PATH.exists():
            raise ApiUnavailable("No fue posible consultar la API ni los metadatos locales.")
        metadata = json.loads(LOCAL_METADATA_PATH.read_text(encoding="utf-8"))
        return {
            "features": metadata["features"],
            "numeric_features": metadata["numeric_features"],
            "categorical_features": metadata["categorical_features"],
            "categories": metadata["categories"],
            "classes": metadata["classes"],
        }


def predict(payload: dict, timeout: float = 10.0) -> dict:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=timeout)
        if response.status_code == 422:
            detail = response.json().get("detail", "Entrada inválida")
            raise ValueError(f"La API rechazó los datos: {detail}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ApiUnavailable(
            f"No fue posible comunicarse con la API en {API_URL}."
        ) from exc

