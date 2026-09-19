from __future__ import annotations

import pandas as pd

from src.data import historical_context


def test_historical_context_uses_most_specific_supported_group() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "municipio_hecho": "PEREIRA",
                "zona_hecho": "URBANA",
                "vehiculo": "MOTOCICLETA",
                "condicion_victima": "CONDUCTOR",
                "dia_semana": "VIERNES",
                "clase_siniestro_modelo": (
                    "CHOQUE CON OTRO VEHICULO" if index < 14 else "ATROPELLO"
                ),
            }
        )
    rows.append(
        {
            "municipio_hecho": "PEREIRA",
            "zona_hecho": "RURAL",
            "vehiculo": "BICICLETA",
            "condicion_victima": "CONDUCTOR",
            "dia_semana": "LUNES",
            "clase_siniestro_modelo": "OTROS",
        }
    )
    frame = pd.DataFrame(rows)
    payload = {
        "municipio_hecho": "PEREIRA",
        "zona_hecho": "URBANA",
        "vehiculo": "MOTOCICLETA",
        "condicion_victima": "CONDUCTOR",
        "dia_semana": "VIERNES",
    }

    result = historical_context(
        frame,
        payload,
        predicted_class="CHOQUE CON OTRO VEHICULO",
    )

    assert result["records"] == 20
    assert result["similarity"] == "ALTA"
    assert len(result["criteria"]) == 5
    assert result["top_class"] == "CHOQUE CON OTRO VEHICULO"
    assert result["aligned"] is True
    assert result["predicted_class_share"] == 0.7


def test_historical_context_relaxes_group_when_exact_sample_is_small() -> None:
    frame = pd.DataFrame(
        [
            {
                "municipio_hecho": "PEREIRA",
                "zona_hecho": "URBANA",
                "vehiculo": "MOTOCICLETA",
                "condicion_victima": "CONDUCTOR",
                "dia_semana": "VIERNES" if index < 4 else "SABADO",
                "clase_siniestro_modelo": "ATROPELLO",
            }
            for index in range(18)
        ]
    )
    payload = {
        "municipio_hecho": "PEREIRA",
        "zona_hecho": "URBANA",
        "vehiculo": "MOTOCICLETA",
        "condicion_victima": "CONDUCTOR",
        "dia_semana": "VIERNES",
    }

    result = historical_context(frame, payload, predicted_class="ATROPELLO")

    assert result["records"] == 18
    assert len(result["criteria"]) == 4
    assert result["aligned"] is True
