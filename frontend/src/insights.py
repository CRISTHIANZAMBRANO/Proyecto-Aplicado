from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd


EARTH_RADIUS_METERS = 6_371_000
UNKNOWN_VALUES = {
    "",
    "-",
    "NAN",
    "NO APLICA",
    "SIN DATO",
    "SIN DATOS",
    "SIN INFORMACION",
}
MOTORCYCLE_VEHICLES = {"MOTOCICLETA", "MOTOCARRO", "MOTO TAXI", "CUATRIMOTO"}


def _mode(series: pd.Series, fallback: str = "SIN INFORMACION") -> str:
    clean = series.dropna().astype(str)
    clean = clean[~clean.str.upper().isin(UNKNOWN_VALUES)]
    modes = clean.mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def _percentage(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def motorcycle_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify unambiguous motorcycle categories.

    The historical label ``MOTO CARRO`` is excluded because it is distinct from both
    ``MOTOCICLETA`` and ``MOTOCARRO`` and cannot be disaggregated reliably.
    """
    vehicles = frame["vehiculo"].fillna("SIN INFORMACION").astype(str).str.upper().str.strip()
    return vehicles.isin(MOTORCYCLE_VEHICLES)


def observed_time_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a usable recorded time.

    The recent source contains 00:00 as a technical default. The historical source
    also has an implausibly large concentration at exactly 00:00, so that value is
    excluded from time-of-day interpretations.
    """
    hour = pd.to_numeric(frame["hora"], errors="coerce")
    return frame.loc[hour.notna() & hour.ne(0)].copy()


def mark_coordinate_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Flag repeated coordinates that represent many different addresses.

    A coordinate reused by at least five records is treated as spatially coarse when
    at least half of those records have different address strings. It remains valid
    for coverage reporting, but is excluded from automatic hotspot recommendations.
    """
    valid = frame.loc[
        frame["coordenadas_validas"].eq(True)
        & frame["latitud"].notna()
        & frame["longitud"].notna()
    ].copy()
    if valid.empty:
        valid["coordenada_agregada"] = pd.Series(dtype=bool)
        return valid

    valid["latitud_6"] = pd.to_numeric(valid["latitud"], errors="coerce").round(6)
    valid["longitud_6"] = pd.to_numeric(valid["longitud"], errors="coerce").round(6)
    addresses = valid.get("direccion_hecho", pd.Series(index=valid.index, dtype="object"))
    valid["direccion_normalizada"] = (
        addresses.fillna("SIN INFORMACION").astype(str).str.strip().str.upper()
    )
    quality = (
        valid.groupby(["latitud_6", "longitud_6"], as_index=False)
        .agg(
            registros=("latitud_6", "size"),
            direcciones=("direccion_normalizada", "nunique"),
        )
    )
    quality["proporcion_direcciones"] = quality["direcciones"] / quality["registros"]
    quality["coordenada_agregada"] = quality["registros"].ge(5) & quality[
        "proporcion_direcciones"
    ].ge(0.5)
    valid = valid.merge(
        quality[
            [
                "latitud_6",
                "longitud_6",
                "registros",
                "direcciones",
                "coordenada_agregada",
            ]
        ],
        on=["latitud_6", "longitud_6"],
        how="left",
        validate="many_to_one",
    )
    return valid


def _neighbor_indices(coordinates_rad: np.ndarray, index: int, radius_m: float) -> np.ndarray:
    latitude = coordinates_rad[:, 0]
    longitude = coordinates_rad[:, 1]
    lat0, lon0 = coordinates_rad[index]
    delta_lat = latitude - lat0
    delta_lon = longitude - lon0
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat0) * np.cos(latitude) * np.sin(delta_lon / 2) ** 2
    )
    distance = 2 * EARTH_RADIUS_METERS * np.arcsin(np.sqrt(haversine))
    return np.flatnonzero(distance <= radius_m)


def cluster_hotspots(
    frame: pd.DataFrame,
    radius_m: int = 150,
    minimum_records: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Detect candidate concentrations with a compact DBSCAN-equivalent routine."""
    mapped = mark_coordinate_quality(frame)
    candidates = mapped.loc[~mapped["coordenada_agregada"]].copy().reset_index(drop=True)
    diagnostics = {
        "valid_coordinates": int(len(mapped)),
        "coarse_coordinates": int(mapped["coordenada_agregada"].sum()),
        "candidate_coordinates": int(len(candidates)),
    }
    if len(candidates) < minimum_records:
        return candidates.assign(cluster_id=-1), pd.DataFrame(), diagnostics

    coordinates = np.radians(
        candidates[["latitud", "longitud"]].astype(float).to_numpy()
    )
    labels = np.full(len(candidates), -1, dtype=int)
    visited = np.zeros(len(candidates), dtype=bool)
    cluster_id = 0

    for index in range(len(candidates)):
        if visited[index]:
            continue
        visited[index] = True
        neighbors = _neighbor_indices(coordinates, index, radius_m)
        if len(neighbors) < minimum_records:
            continue

        labels[index] = cluster_id
        queue = deque(int(value) for value in neighbors if int(value) != index)
        queued = set(queue)
        while queue:
            neighbor = queue.popleft()
            if not visited[neighbor]:
                visited[neighbor] = True
                neighbor_neighbors = _neighbor_indices(coordinates, neighbor, radius_m)
                if len(neighbor_neighbors) >= minimum_records:
                    for value in neighbor_neighbors:
                        value = int(value)
                        if value not in queued:
                            queue.append(value)
                            queued.add(value)
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
        cluster_id += 1

    candidates["cluster_id"] = labels
    clustered = candidates.loc[candidates["cluster_id"].ge(0)].copy()
    if clustered.empty:
        return candidates, pd.DataFrame(), diagnostics

    summaries = []
    for current_cluster, group in clustered.groupby("cluster_id"):
        modelable = group["clase_siniestro_modelo"].dropna()
        dominant_class = _mode(modelable, "NO MODELABLE")
        dominant_share = (
            _percentage(int(modelable.eq(dominant_class).sum()), len(modelable))
            if not modelable.empty
            else 0.0
        )
        motorcycle_share = float(motorcycle_mask(group).mean())
        pedestrian_share = float(group["condicion_victima"].eq("PEATON").mean())
        time_rows = observed_time_records(group)
        night_share = (
            float(time_rows["franja_horaria"].isin(["NOCHE", "MADRUGADA"]).mean())
            if not time_rows.empty
            else np.nan
        )
        addresses = group["direccion_hecho"].fillna("").astype(str)
        address_text = " ".join(addresses.str.upper().tolist())
        if any(keyword in address_text for keyword in ("GLORIETA", "ROTONDA", "REDOMA", "ROMBOY")):
            detected_context = "Rotonda o glorieta"
        elif any(
            keyword in address_text
            for keyword in ("INTERSECCION", "ESQUINA", " CRUCE ", " CON CALLE ", " CON CARRERA ")
        ):
            detected_context = "Intersección"
        else:
            detected_context = "No especificado"

        location = _mode(group["barrio_hecho"], _mode(group["direccion_hecho"], "Sector sin nombre"))
        summaries.append(
            {
                "cluster_id": int(current_cluster),
                "registros": int(len(group)),
                "latitud": float(group["latitud"].median()),
                "longitud": float(group["longitud"].median()),
                "municipio": _mode(group["municipio_hecho"]),
                "sector": location,
                "clase_predominante": dominant_class,
                "participacion_clase": dominant_share,
                "participacion_moto": motorcycle_share,
                "participacion_peaton": pedestrian_share,
                "participacion_nocturna": night_share,
                "contexto_detectado": detected_context,
                "fecha_inicial": group["fecha_hecho"].min(),
                "fecha_final": group["fecha_hecho"].max(),
            }
        )

    summary = pd.DataFrame(summaries).sort_values(
        ["registros", "municipio"], ascending=[False, True]
    )
    summary["prioridad_visual"] = range(1, len(summary) + 1)
    summary["etiqueta"] = summary.apply(
        lambda row: (
            f"Candidato {row['prioridad_visual']} · {row['municipio']} · "
            f"{row['sector']} · {row['registros']} registros"
        ),
        axis=1,
    )
    rank_by_cluster = summary.set_index("cluster_id")["prioridad_visual"].to_dict()
    clustered["prioridad_visual"] = clustered["cluster_id"].map(rank_by_cluster)
    diagnostics["hotspots"] = int(len(summary))
    diagnostics["clustered_records"] = int(len(clustered))
    return clustered, summary.reset_index(drop=True), diagnostics


def hotspot_recommendation(group: pd.DataFrame, site_context: str) -> dict:
    modelable = group["clase_siniestro_modelo"].dropna()
    dominant_class = _mode(modelable, "OTROS")
    actions = [
        "Validar la precisión de las coordenadas y realizar una inspección del sitio.",
        "Contrastar la concentración con volúmenes de tránsito, geometría y condiciones operativas.",
    ]

    if dominant_class == "ATROPELLO":
        focus = "Protección de peatones y reducción de conflictos vehículo-peatón"
        actions.extend(
            [
                "Evaluar cruces, continuidad peatonal, visibilidad, iluminación y velocidades de aproximación.",
                "Revisar si los tiempos y dispositivos de cruce protegen adecuadamente a los peatones.",
            ]
        )
    elif dominant_class == "CHOQUE CON OTRO VEHICULO":
        focus = "Gestión de conflictos entre vehículos"
        actions.extend(
            [
                "Revisar prioridades de paso, demarcación, canalización y movimientos conflictivos.",
                "Evaluar medidas de gestión de velocidad y control de maniobras.",
            ]
        )
    elif dominant_class == "CHOQUE CON OBJETO FIJO":
        focus = "Condiciones del corredor y objetos laterales"
        actions.extend(
            [
                "Revisar delineación, iluminación, obstáculos laterales y zonas de recuperación.",
                "Evaluar velocidad de operación y visibilidad del trazado.",
            ]
        )
    else:
        focus = "Caracterización adicional del punto"
        actions.append(
            "Revisar los registros individuales antes de seleccionar una intervención específica."
        )

    if site_context == "Rotonda o glorieta":
        actions.extend(
            [
                "Revisar ángulos de entrada, control de velocidad, señalización de prioridad y uso de carriles.",
                "Evaluar técnicamente la viabilidad de semaforización u otros controles; la concentración por sí sola no justifica instalarlos.",
            ]
        )
    elif site_context == "Intersección":
        actions.append(
            "Evaluar controles de tránsito, incluida la viabilidad de semaforización, mediante un estudio operacional y de seguridad."
        )
    elif site_context == "Tramo vial":
        actions.append(
            "Revisar consistencia del límite de velocidad, iluminación, demarcación y accesos del tramo."
        )

    motorcycle_share = float(motorcycle_mask(group).mean())
    pedestrian_share = float(group["condicion_victima"].eq("PEATON").mean())
    if motorcycle_share >= 0.5:
        actions.append(
            "Incluir visibilidad de motociclistas, estado de la superficie y conflictos de giro en la revisión."
        )
    if pedestrian_share >= 0.4 and dominant_class != "ATROPELLO":
        actions.append("Incluir condiciones de cruce y accesibilidad peatonal en la auditoría.")

    return {
        "focus": focus,
        "dominant_class": dominant_class,
        "actions": list(dict.fromkeys(actions)),
        "disclaimer": (
            "Recomendación exploratoria basada en registros históricos. No prescribe una obra ni "
            "reemplaza un estudio de tránsito, una auditoría de seguridad vial o una visita de campo."
        ),
    }


def build_myths(frame: pd.DataFrame) -> list[dict]:
    total = len(frame)
    if total == 0:
        return []

    motorcycle = motorcycle_mask(frame)
    motorcycle_share = float(motorcycle.mean())

    valid_days = frame["dia_semana"].dropna()
    weekend_share = (
        float(valid_days.isin(["SABADO", "DOMINGO"]).mean())
        if not valid_days.empty
        else 0.0
    )

    male_share = float(frame["sexo"].eq("MASCULINO").mean())
    pereira_share = float(frame["municipio_hecho"].eq("PEREIRA").mean())

    age_counts = frame["rango_edad"].value_counts()
    top_age = str(age_counts.index[0]) if not age_counts.empty else "SIN INFORMACION"
    top_age_share = _percentage(int(age_counts.iloc[0]), int(age_counts.sum())) if not age_counts.empty else 0.0

    known_road = frame.loc[
        ~frame["estado_via"].fillna("SIN INFORMACION").isin(UNKNOWN_VALUES),
        "estado_via",
    ]
    adverse_road = known_road.isin(
        ["REGULAR", "MALO", "CON HUECOS", "FISURADA", "HUNDIMIENTO", "RIZADA", "ROCAS EN LA VIA"]
    )
    adverse_share = float(adverse_road.mean()) if not known_road.empty else 0.0
    road_missing = 1 - _percentage(len(known_road), total)

    known_weather = frame.loc[
        ~frame["condicion_lugar"].fillna("SIN INFORMACION").isin(UNKNOWN_VALUES),
        "condicion_lugar",
    ]
    rain_share = float(known_weather.eq("LLUVIA").mean()) if not known_weather.empty else 0.0
    weather_missing = 1 - _percentage(len(known_weather), total)

    return [
        {
            "claim": "La mayoría de los registros fatales está asociada a motocicletas.",
            "verdict": "RESPALDADO" if motorcycle_share > 0.5 else "DESMENTIDO",
            "metric": f"{motorcycle_share:.1%}",
            "evidence": (
                f"{int(motorcycle.sum())} de {total} registros tienen una categoría inequívoca de motocicleta. "
                "Es un grupo importante, pero solo constituye mayoría si supera el 50%."
            ),
            "limitation": (
                "El vehículo corresponde al asociado a la víctima y el consolidado solo contiene casos fatales. "
                "La etiqueta histórica ambigua 'MOTO CARRO' no se cuenta como motocicleta."
            ),
        },
        {
            "claim": "La edad determina una mayor probabilidad de que ocurra un siniestro.",
            "verdict": "NO DEMOSTRABLE",
            "metric": f"Grupo más frecuente: {top_age} ({top_age_share:.1%})",
            "evidence": "Podemos describir las edades presentes en los registros, pero no calcular probabilidad por edad.",
            "limitation": "Falta el número de personas expuestas o conductores por grupo de edad y no todos los registros corresponden a conductores.",
        },
        {
            "claim": "La mayoría de los registros ocurre durante el fin de semana.",
            "verdict": "RESPALDADO" if weekend_share > 0.5 else "DESMENTIDO",
            "metric": f"{weekend_share:.1%}",
            "evidence": "La proporción combina sábado y domingo sobre los registros con día disponible.",
            "limitation": "Describe la distribución de casos fatales; no controla por volumen de viajes de cada día.",
        },
        {
            "claim": "Los hombres predominan entre las víctimas registradas.",
            "verdict": "RESPALDADO" if male_share > 0.5 else "DESMENTIDO",
            "metric": f"{male_share:.1%}",
            "evidence": f"{int(frame['sexo'].eq('MASCULINO').sum())} de {total} registros corresponden a sexo masculino.",
            "limitation": "Predominio descriptivo no significa que el sexo sea una causa del siniestro.",
        },
        {
            "claim": "La mayoría de los casos con estado de vía conocido ocurrió en vías adversas.",
            "verdict": "RESPALDADO" if adverse_share > 0.5 else "DESMENTIDO",
            "metric": f"{adverse_share:.1%} de los conocidos",
            "evidence": "Se agruparon como adversos los estados regular, malo, con huecos, fisurado, hundimiento, rizado y rocas.",
            "limitation": f"El estado de la vía no está informado en {road_missing:.1%} de los registros y no prueba causalidad.",
        },
        {
            "claim": "La lluvia aparece en la mayoría de los registros con condición del lugar conocida.",
            "verdict": "RESPALDADO" if rain_share > 0.5 else "DESMENTIDO",
            "metric": f"{rain_share:.1%} de los conocidos",
            "evidence": "La comparación usa únicamente registros con condición del lugar informada.",
            "limitation": f"La condición del lugar no está informada en {weather_missing:.1%} de los registros.",
        },
        {
            "claim": "Pereira concentra la mayoría de los registros del consolidado.",
            "verdict": "RESPALDADO" if pereira_share > 0.5 else "DESMENTIDO",
            "metric": f"{pereira_share:.1%}",
            "evidence": f"{int(frame['municipio_hecho'].eq('PEREIRA').sum())} de {total} registros corresponden a Pereira.",
            "limitation": "La fuente reciente 2023-2025 solo cubre Pereira; no equivale a una comparación de riesgo entre municipios.",
        },
    ]
