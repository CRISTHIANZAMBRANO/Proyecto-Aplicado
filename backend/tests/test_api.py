from __future__ import annotations

import math

from fastapi.testclient import TestClient

from app.main import app


VALID_PAYLOAD = {
    "edad": 35,
    "hora": 18.0,
    "sexo": "MASCULINO",
    "zona_hecho": "URBANA",
    "vehiculo": "MOTOCICLETA",
    "condicion_victima": "CONDUCTOR",
    "dia_semana": "VIERNES",
    "municipio_hecho": "PEREIRA",
    "mes_hecho": 8,
    "tipo_servicio_vehiculo": "PARTICULAR",
    "escenario_hecho": "VIA PUBLICA",
    "condicion_lugar": "SECO",
    "estado_via": "BUENO",
}


def test_health_and_metadata() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        options = client.get("/model/options")
        assert options.status_code == 200
        assert len(options.json()["features"]) == 13
        assert "PEREIRA" in options.json()["categories"]["municipio_hecho"]


def test_prediction_contract() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200, response.text
        result = response.json()

    assert result["prediction"] in result["probabilities"]
    assert result["experimental"] is True
    assert result["metric_context"]["target_met"] is False
    assert result["recommendation"]["actions"]
    assert result["recommendation"]["validation_steps"]
    assert result["decision_support"]["runner_up_class"] in result["probabilities"]
    assert result["decision_support"]["runner_up_class"] != result["prediction"]
    assert result["decision_support"]["signal_strength"] in {"ALTA", "MEDIA", "BAJA"}
    assert math.isclose(
        result["decision_support"]["probability_margin"],
        result["confidence"] - result["decision_support"]["runner_up_probability"],
        abs_tol=1e-5,
    )
    assert math.isclose(sum(result["probabilities"].values()), 1.0, abs_tol=1e-5)
    assert 0 <= result["confidence"] <= 1
    assert result["relative_index"] >= 0


def test_invalid_numeric_range_returns_422() -> None:
    invalid = {**VALID_PAYLOAD, "edad": -1}
    with TestClient(app) as client:
        response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_unknown_category_is_reported_not_crashed() -> None:
    unknown = {**VALID_PAYLOAD, "estado_via": "CATEGORIA NUEVA"}
    with TestClient(app) as client:
        response = client.post("/predict", json=unknown)
        assert response.status_code == 200
        warnings = response.json()["warnings"]
    assert any("categoría desconocida" in warning for warning in warnings)
