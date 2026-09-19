from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import DATA_PATH


@st.cache_data(show_spinner="Cargando datos consolidados...")
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el consolidado en {DATA_PATH}. Configure DATA_PATH."
        )
    frame = pd.read_excel(DATA_PATH, sheet_name="Datos_consolidados")
    frame["fecha_hecho"] = pd.to_datetime(frame["fecha_hecho"], errors="coerce")
    frame["periodo_mes"] = frame["fecha_hecho"].dt.to_period("M").astype("string")
    return frame


def filter_frame(
    frame: pd.DataFrame,
    years: list[int],
    municipalities: list[str],
    zones: list[str],
    vehicles: list[str],
    classes: list[str] | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if years:
        filtered = filtered[filtered["anio_hecho"].isin(years)]
    if municipalities:
        filtered = filtered[filtered["municipio_hecho"].isin(municipalities)]
    if zones:
        filtered = filtered[filtered["zona_hecho"].isin(zones)]
    if vehicles:
        filtered = filtered[filtered["vehiculo"].isin(vehicles)]
    if classes:
        filtered = filtered[filtered["clase_siniestro_modelo"].isin(classes)]
    return filtered


def sorted_options(frame: pd.DataFrame, column: str) -> list:
    values = frame[column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


HISTORICAL_MATCH_PLANS = [
    ("municipio_hecho", "zona_hecho", "vehiculo", "condicion_victima", "dia_semana"),
    ("municipio_hecho", "zona_hecho", "vehiculo", "condicion_victima"),
    ("municipio_hecho", "vehiculo", "condicion_victima"),
    ("municipio_hecho", "vehiculo"),
    ("municipio_hecho", "condicion_victima"),
    ("municipio_hecho", "zona_hecho"),
    ("municipio_hecho",),
]

HISTORICAL_LABELS = {
    "municipio_hecho": "municipio",
    "zona_hecho": "zona",
    "vehiculo": "vehículo",
    "condicion_victima": "condición de la víctima",
    "dia_semana": "día de la semana",
}


def historical_context(
    frame: pd.DataFrame,
    payload: dict,
    predicted_class: str,
    minimum_records: int = 15,
) -> dict:
    """Describe casos registrados comparables sin convertirlos en riesgo de ocurrencia."""
    modelable = frame.dropna(subset=["clase_siniestro_modelo"]).copy()
    selected = None
    selected_fields: tuple[str, ...] = ()

    for fields in HISTORICAL_MATCH_PLANS:
        mask = pd.Series(True, index=modelable.index)
        for field in fields:
            mask &= modelable[field].astype("string").eq(str(payload[field]))
        candidate = modelable.loc[mask]
        if len(candidate) >= minimum_records:
            selected = candidate
            selected_fields = fields
            break

    if selected is None:
        selected = modelable
        selected_fields = ()

    counts = selected["clase_siniestro_modelo"].value_counts()
    shares = selected["clase_siniestro_modelo"].value_counts(normalize=True)
    distribution = pd.DataFrame(
        {
            "clase": counts.index,
            "registros": counts.values,
            "participacion_historica": shares.reindex(counts.index).values,
        }
    )
    top_class = str(counts.index[0])
    predicted_share = float(shares.get(predicted_class, 0.0))
    criteria = [
        {
            "field": field,
            "label": HISTORICAL_LABELS[field],
            "value": str(payload[field]),
        }
        for field in selected_fields
    ]
    if selected_fields:
        scope = ", ".join(f"{item['label']}: {item['value']}" for item in criteria)
        similarity = "ALTA" if len(selected_fields) >= 4 else "MEDIA" if len(selected_fields) >= 2 else "AMPLIA"
    else:
        scope = "todo el histórico modelable"
        similarity = "GENERAL"

    return {
        "records": int(len(selected)),
        "criteria": criteria,
        "scope": scope,
        "similarity": similarity,
        "distribution": distribution,
        "top_class": top_class,
        "predicted_class_share": predicted_share,
        "aligned": top_class == predicted_class,
    }
