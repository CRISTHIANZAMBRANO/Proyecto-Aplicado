from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import filter_frame, load_data, sorted_options
from src.insights import (
    UNKNOWN_VALUES,
    build_myths,
    motorcycle_mask,
    observed_time_records,
)
from src.ui import (
    COLORS,
    configure_page,
    page_header,
    render_footer,
    render_verdict_card,
    section_header,
    style_chart,
)


MONTH_ORDER = [
    "ENERO",
    "FEBRERO",
    "MARZO",
    "ABRIL",
    "MAYO",
    "JUNIO",
    "JULIO",
    "AGOSTO",
    "SEPTIEMBRE",
    "OCTUBRE",
    "NOVIEMBRE",
    "DICIEMBRE",
]

configure_page("Dashboard EDA", "📊")
data = load_data()
context = st.session_state.get("analysis_context", {})
available_years = sorted(int(value) for value in data["anio_hecho"].dropna().unique())

page_header(
    "Módulo 1 · análisis descriptivo",
    "Siniestralidad vial fatal",
    "Resumen ejecutivo, análisis detallado y verificación de afirmaciones populares con el consolidado disponible.",
    badge=f"{len(data):,} registros · {min(available_years)}–{max(available_years)}".replace(",", "."),
)

st.sidebar.header("Filtros globales")
years = st.sidebar.multiselect("Año", available_years, placeholder="Todos los años")
municipalities = st.sidebar.multiselect(
    "Municipio",
    sorted_options(data, "municipio_hecho"),
    default=context.get("municipalities", []),
    placeholder="Todos los municipios",
)
zones = st.sidebar.multiselect(
    "Zona",
    sorted_options(data, "zona_hecho"),
    default=context.get("zones", []),
    placeholder="Todas las zonas",
)
vehicles = st.sidebar.multiselect(
    "Vehículo asociado",
    sorted_options(data, "vehiculo"),
    default=context.get("vehicles", []),
    placeholder="Todos los vehículos",
)
classes = st.sidebar.multiselect(
    "Clase de siniestro",
    sorted_options(data.dropna(subset=["clase_siniestro_modelo"]), "clase_siniestro_modelo"),
    placeholder="Todas las clases",
)

filtered = filter_frame(data, years, municipalities, zones, vehicles, classes)
if filtered.empty:
    st.warning("La combinación de filtros no devuelve registros.")
    st.stop()

filter_parts = []
if years:
    filter_parts.append(f"años: {', '.join(map(str, years))}")
if municipalities:
    filter_parts.append(f"municipios: {', '.join(municipalities)}")
if zones:
    filter_parts.append(f"zonas: {', '.join(zones)}")
if vehicles:
    filter_parts.append(f"vehículos: {', '.join(vehicles)}")
if classes:
    filter_parts.append(f"clases: {', '.join(classes)}")
st.caption(
    "Vista actual: " + (" · ".join(filter_parts) if filter_parts else "todo el consolidado")
)

if context:
    st.info(
        "El municipio, la zona y el vehículo fueron precargados desde el simulador. "
        "Los filtros pueden modificarse para contrastar el escenario."
    )

summary_tab, detail_tab, myths_tab = st.tabs(
    ["Resumen ejecutivo", "Análisis detallado", "Mitos vs. datos"]
)

with summary_tab:
    valid_coordinates = int(filtered["coordenadas_validas"].eq(True).sum())
    coordinate_coverage = valid_coordinates / len(filtered)
    motorcycle_share = motorcycle_mask(filtered).mean()
    top_municipality = str(filtered["municipio_hecho"].mode().iloc[0])

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Registros fatales",
        f"{len(filtered):,}".replace(",", "."),
        help="Cada fila representa una víctima fatal incluida en el consolidado.",
    )
    metric_columns[1].metric(
        "Municipio con más registros",
        top_municipality.title(),
        f"{filtered['municipio_hecho'].eq(top_municipality).mean():.1%} del filtro",
    )
    metric_columns[2].metric(
        "Asociados a motocicleta",
        f"{motorcycle_share:.1%}",
        help="Participación de categorías inequívocas de motocicleta; excluye la etiqueta histórica ambigua 'MOTO CARRO'.",
    )
    metric_columns[3].metric(
        "Cobertura geográfica",
        f"{coordinate_coverage:.1%}",
        help="Coordenadas dentro de rangos válidos; no garantiza precisión a nivel de intersección.",
    )

    st.write("")
    left, right = st.columns([1.55, 1])
    with left:
        with st.container(border=True):
            annual = (
                filtered.dropna(subset=["anio_hecho"])
                .assign(anio=lambda frame: frame["anio_hecho"].astype(int))
                .groupby("anio", as_index=False)
                .size()
                .rename(columns={"size": "registros"})
            )
            if not annual.empty:
                complete_years = pd.DataFrame(
                    {"anio": range(int(annual["anio"].min()), int(annual["anio"].max()) + 1)}
                )
                annual = complete_years.merge(annual, on="anio", how="left")
            fig = px.line(
                annual,
                x="anio",
                y="registros",
                markers=True,
                title="Evolución anual de registros",
                color_discrete_sequence=[COLORS["blue"]],
            )
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title="Año", dtick=2)
            fig.update_yaxes(title="Registros")
            st.plotly_chart(style_chart(fig, 400), width="stretch")
            st.caption("Los vacíos representan años sin datos disponibles, como 2022.")

    with right:
        with st.container(border=True):
            class_counts = (
                filtered["clase_siniestro_modelo"]
                .fillna("NO MODELABLE")
                .value_counts()
                .rename_axis("clase")
                .reset_index(name="registros")
                .sort_values("registros")
            )
            fig = px.bar(
                class_counts,
                x="registros",
                y="clase",
                orientation="h",
                title="Registros por clase de siniestro",
                color="registros",
                color_continuous_scale=["#CFE5F8", COLORS["blue"]],
                text_auto=True,
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="Registros")
            fig.update_yaxes(title="")
            st.plotly_chart(style_chart(fig, 400), width="stretch")

    left, right = st.columns([1, 1.55])
    with left:
        with st.container(border=True):
            victim = (
                filtered["condicion_victima"]
                .fillna("SIN INFORMACION")
                .value_counts()
                .rename_axis("condicion")
                .reset_index(name="registros")
            )
            fig = px.pie(
                victim,
                names="condicion",
                values="registros",
                hole=0.62,
                title="Condición de la víctima",
                color_discrete_sequence=[
                    COLORS["navy"],
                    COLORS["blue"],
                    COLORS["teal"],
                    COLORS["orange"],
                ],
            )
            fig.update_traces(textposition="outside", textinfo="percent")
            st.plotly_chart(style_chart(fig, 410), width="stretch")

    with right:
        with st.container(border=True):
            municipality = (
                filtered.assign(
                    motocicleta=motorcycle_mask(filtered)
                )
                .groupby("municipio_hecho", as_index=False)
                .agg(
                    registros=("registro_id", "size"),
                    cobertura_geografica=("coordenadas_validas", "mean"),
                    participacion_moto=("motocicleta", "mean"),
                )
                .sort_values("registros", ascending=False)
                .rename(columns={"municipio_hecho": "Municipio"})
            )
            municipality["Cobertura geográfica"] = municipality[
                "cobertura_geografica"
            ].map(lambda value: f"{value:.1%}")
            municipality["Participación moto"] = municipality[
                "participacion_moto"
            ].map(lambda value: f"{value:.1%}")
            municipality = municipality.rename(columns={"registros": "Registros"})
            st.markdown("#### Registros por municipio")
            st.dataframe(
                municipality[
                    ["Municipio", "Registros", "Participación moto", "Cobertura geográfica"]
                ],
                width="stretch",
                hide_index=True,
                height=350,
            )

with detail_tab:
    section_header(
        "Desglose",
        "Comportamiento temporal y condiciones registradas",
        "Las visualizaciones responden a los filtros globales.",
    )
    left, right = st.columns([1.55, 1])
    with left:
        with st.container(border=True):
            month_class = (
                filtered.dropna(subset=["mes_nombre"])
                .assign(
                    clase=lambda frame: frame["clase_siniestro_modelo"].fillna("NO MODELABLE")
                )
                .groupby(["mes_nombre", "clase"], as_index=False)
                .size()
                .rename(columns={"size": "registros"})
            )
            month_class["mes_nombre"] = pd.Categorical(
                month_class["mes_nombre"], categories=MONTH_ORDER, ordered=True
            )
            month_class = month_class.sort_values("mes_nombre")
            fig = px.bar(
                month_class,
                x="mes_nombre",
                y="registros",
                color="clase",
                title="Registros por mes y clase",
                color_discrete_sequence=[
                    COLORS["navy"],
                    COLORS["blue"],
                    COLORS["teal"],
                    COLORS["blue_light"],
                    COLORS["orange"],
                ],
            )
            fig.update_layout(barmode="stack")
            fig.update_xaxes(title="", tickangle=-30)
            fig.update_yaxes(title="Registros")
            st.plotly_chart(style_chart(fig, 430), width="stretch")

    with right:
        with st.container(border=True):
            time_data = observed_time_records(filtered)
            time_order = ["MADRUGADA", "MANANA", "TARDE", "NOCHE"]
            time_counts = (
                time_data["franja_horaria"]
                .value_counts()
                .reindex(time_order, fill_value=0)
                .rename_axis("franja")
                .reset_index(name="registros")
            )
            fig = px.bar(
                time_counts,
                x="registros",
                y="franja",
                orientation="h",
                title="Registros por franja horaria observada",
                color="registros",
                color_continuous_scale=["#CFE5F8", COLORS["blue"]],
                text_auto=True,
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="Registros")
            fig.update_yaxes(title="")
            st.plotly_chart(style_chart(fig, 355), width="stretch")
            st.caption(
                "Se excluyen horas 00:00 por su uso como valor técnico repetido. "
                f"Registros utilizables: {len(time_data):,}.".replace(",", ".")
            )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            road = filtered["estado_via"].fillna("SIN INFORMACION")
            known_road = road.loc[~road.isin(UNKNOWN_VALUES)]
            road_counts = (
                known_road.value_counts()
                .rename_axis("estado")
                .reset_index(name="registros")
            )
            fig = px.pie(
                road_counts,
                names="estado",
                values="registros",
                hole=0.62,
                title="Estado de la vía cuando fue informado",
                color_discrete_sequence=[
                    COLORS["navy"],
                    COLORS["blue"],
                    COLORS["teal"],
                    COLORS["orange"],
                    COLORS["red"],
                ],
            )
            st.plotly_chart(style_chart(fig, 405), width="stretch")
            st.caption(
                f"Sin información o no aplicable: {1 - len(known_road) / len(filtered):.1%}."
            )

    with right:
        with st.container(border=True):
            age_order = [
                "0 A 14",
                "15 A 17",
                "18 A 24",
                "25 A 34",
                "35 A 44",
                "45 A 54",
                "55 A 64",
                "65 O MAS",
            ]
            age = (
                filtered["rango_edad"]
                .value_counts()
                .reindex(age_order, fill_value=0)
                .rename_axis("rango")
                .reset_index(name="registros")
            )
            fig = px.bar(
                age,
                x="rango",
                y="registros",
                title="Víctimas registradas por rango de edad",
                color="registros",
                color_continuous_scale=["#CFE5F8", COLORS["navy"]],
                text_auto=True,
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="", tickangle=-25)
            fig.update_yaxes(title="Registros")
            st.plotly_chart(style_chart(fig, 405), width="stretch")

    with st.container(border=True):
        st.markdown("#### Detalle de registros")
        visible_columns = [
            "fecha_hecho",
            "municipio_hecho",
            "barrio_hecho",
            "zona_hecho",
            "vehiculo",
            "condicion_victima",
            "clase_siniestro_modelo",
            "estado_via",
        ]
        detail = filtered[visible_columns].sort_values("fecha_hecho", ascending=False).head(500)
        st.dataframe(detail, width="stretch", hide_index=True, height=430)
        st.caption("Se muestran hasta 500 registros del filtro actual.")

with myths_tab:
    section_header(
        "Verificación descriptiva",
        "Mitos vs. datos",
        "Cada afirmación se evalúa sobre el filtro actual. 'No demostrable' indica que falta un denominador o diseño causal.",
    )
    st.markdown(
        """
        <div class="method-note">
        <strong>Cómo leer esta sección:</strong> respaldado o desmentido se refiere únicamente
        a la distribución del consolidado fatal. No significa causalidad ni riesgo individual.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    myths = build_myths(filtered)
    left, right = st.columns(2)
    for index, myth in enumerate(myths):
        with left if index % 2 == 0 else right:
            render_verdict_card(myth)

render_footer()
