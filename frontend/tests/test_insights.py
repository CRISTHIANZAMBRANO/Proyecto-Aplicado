from __future__ import annotations

import pandas as pd

from src.insights import (
    build_myths,
    cluster_hotspots,
    hotspot_recommendation,
    mark_coordinate_quality,
    motorcycle_mask,
)


def test_coordinate_quality_flags_reused_coordinate_with_distinct_addresses() -> None:
    frame = pd.DataFrame(
        {
            "coordenadas_validas": [True] * 7,
            "latitud": [4.810000] * 5 + [4.82, 4.82],
            "longitud": [-75.690000] * 5 + [-75.70, -75.70],
            "direccion_hecho": [f"CALLE {index}" for index in range(5)]
            + ["MISMA DIRECCION", "MISMA DIRECCION"],
        }
    )

    result = mark_coordinate_quality(frame)

    assert result.loc[result["latitud_6"].eq(4.81), "coordenada_agregada"].all()
    assert not result.loc[result["latitud_6"].eq(4.82), "coordenada_agregada"].any()


def test_motorcycle_mask_excludes_ambiguous_historical_label() -> None:
    frame = pd.DataFrame(
        {"vehiculo": ["MOTOCICLETA", "MOTOCARRO", "MOTO CARRO", "AUTOMOVIL"]}
    )

    result = motorcycle_mask(frame)

    assert result.tolist() == [True, True, False, False]


def _hotspot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "registro_id": [1, 2, 3, 4],
            "coordenadas_validas": [True] * 4,
            "latitud": [4.81000, 4.81025, 4.81050, 4.83000],
            "longitud": [-75.69000, -75.69020, -75.69040, -75.72000],
            "direccion_hecho": [
                "GLORIETA A",
                "GLORIETA A",
                "GLORIETA A",
                "CALLE LEJANA",
            ],
            "barrio_hecho": ["CENTRO", "CENTRO", "CENTRO", "OTRO"],
            "municipio_hecho": ["PEREIRA"] * 4,
            "clase_siniestro_modelo": [
                "CHOQUE CON OTRO VEHICULO",
                "CHOQUE CON OTRO VEHICULO",
                "ATROPELLO",
                "OTROS",
            ],
            "vehiculo": ["MOTOCICLETA", "MOTOCICLETA", "AUTOMOVIL", "AUTOMOVIL"],
            "condicion_victima": ["CONDUCTOR", "CONDUCTOR", "PEATON", "CONDUCTOR"],
            "hora": [8.0, 9.0, 10.0, 11.0],
            "franja_horaria": ["MANANA", "MANANA", "MANANA", "MANANA"],
            "fecha_hecho": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]
            ),
        }
    )


def test_cluster_hotspots_detects_nearby_group() -> None:
    clustered, summary, diagnostics = cluster_hotspots(
        _hotspot_frame(), radius_m=150, minimum_records=3
    )

    assert diagnostics["hotspots"] == 1
    assert len(clustered) == 3
    assert len(summary) == 1
    assert summary.iloc[0]["registros"] == 3
    assert summary.iloc[0]["contexto_detectado"] == "Rotonda o glorieta"


def test_roundabout_recommendation_is_conditional() -> None:
    frame = _hotspot_frame().iloc[:3]

    result = hotspot_recommendation(frame, "Rotonda o glorieta")
    joined = " ".join(result["actions"]).lower()

    assert "semaforización" in joined
    assert "no justifica" in joined
    assert "no prescribe" in result["disclaimer"].lower()


def test_myths_distinguish_description_from_probability() -> None:
    frame = pd.DataFrame(
        {
            "vehiculo": ["MOTOCICLETA", "AUTOMOVIL", "BICICLETA", "AUTOMOVIL"],
            "dia_semana": ["LUNES", "MARTES", "SABADO", "DOMINGO"],
            "sexo": ["MASCULINO", "MASCULINO", "FEMENINO", "MASCULINO"],
            "municipio_hecho": ["PEREIRA", "PEREIRA", "DOSQUEBRADAS", "PEREIRA"],
            "rango_edad": ["18 A 24", "18 A 24", "65 O MAS", "65 O MAS"],
            "estado_via": ["BUENO", "BUENO", "REGULAR", "SIN INFORMACION"],
            "condicion_lugar": ["NORMAL", "NORMAL", "LLUVIA", "SIN INFORMACION"],
        }
    )

    myths = build_myths(frame)
    by_claim = {item["claim"]: item for item in myths}

    motorcycle = by_claim[
        "La mayoría de los registros fatales está asociada a motocicletas."
    ]
    age = by_claim[
        "La edad determina una mayor probabilidad de que ocurra un siniestro."
    ]
    assert motorcycle["verdict"] == "DESMENTIDO"
    assert age["verdict"] == "NO DEMOSTRABLE"
