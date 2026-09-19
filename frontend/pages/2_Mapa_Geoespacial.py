from __future__ import annotations

from html import escape

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from src.data import filter_frame, load_data, sorted_options
from src.insights import cluster_hotspots, hotspot_recommendation
from src.ui import (
    COLORS,
    configure_page,
    page_header,
    render_footer,
    section_header,
    style_chart,
)


configure_page("Mapa geoespacial", "🗺️")
data = load_data()
context = st.session_state.get("analysis_context", {})
available_years = sorted(int(value) for value in data["anio_hecho"].dropna().unique())

page_header(
    "Módulo 2 · priorización territorial",
    "Concentraciones y acciones por validar",
    "Explore la distribución espacial, detecte concentraciones candidatas y convierta cada hallazgo en una revisión técnica concreta.",
    badge="Mapa descriptivo + recomendaciones basadas en reglas transparentes",
)

st.sidebar.header("Filtros del mapa")
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

st.sidebar.divider()
st.sidebar.subheader("Capas y detección")
show_heatmap = st.sidebar.toggle("Mapa de calor", value=True)
show_markers = st.sidebar.toggle("Puntos agrupados", value=False)
show_hotspots = st.sidebar.toggle("Concentraciones candidatas", value=True)
radius_m = st.sidebar.select_slider(
    "Radio de proximidad",
    options=[100, 150, 200],
    value=150,
    help="Distancia máxima usada para relacionar registros vecinos.",
)
minimum_records = st.sidebar.selectbox(
    "Mínimo de registros",
    options=[3, 4, 5],
    index=0,
    help="Cantidad mínima para considerar una concentración candidata.",
)

if context:
    st.info(
        "El municipio, la zona y el vehículo fueron precargados desde el simulador. "
        "Puede modificarlos para ampliar o restringir el contexto."
    )

filtered = filter_frame(data, years, municipalities, zones, vehicles, classes)
if filtered.empty:
    st.warning("La combinación de filtros no devuelve registros.")
    st.stop()

mapped = filtered.loc[
    filtered["coordenadas_validas"].eq(True)
    & filtered["latitud"].notna()
    & filtered["longitud"].notna()
].copy()
if mapped.empty:
    st.warning("No existen coordenadas válidas para la combinación seleccionada.")
    st.stop()

clustered, hotspots, diagnostics = cluster_hotspots(
    filtered,
    radius_m=int(radius_m),
    minimum_records=int(minimum_records),
)

metrics = st.columns(4)
metrics[0].metric("Registros filtrados", f"{len(filtered):,}".replace(",", "."))
metrics[1].metric(
    "Coordenadas válidas",
    f"{diagnostics['valid_coordinates']:,}".replace(",", "."),
    f"{diagnostics['valid_coordinates'] / len(filtered):.1%} del filtro",
)
metrics[2].metric(
    "Puntos analizables",
    f"{diagnostics['candidate_coordinates']:,}".replace(",", "."),
    help="Excluye coordenadas repetidas que parecen representar muchos domicilios distintos.",
)
metrics[3].metric(
    "Concentraciones candidatas",
    f"{len(hotspots):,}".replace(",", "."),
    help=f"Radio de {radius_m} m y al menos {minimum_records} registros.",
)

st.markdown(
    f"""
    <div class="method-note">
    <strong>Control de calidad espacial:</strong> se detectaron
    <strong>{diagnostics['coarse_coordinates']:,}</strong> registros con coordenadas válidas
    pero posiblemente agregadas o aproximadas. Se visualizan en la densidad general, pero se
    excluyen de las recomendaciones automáticas para evitar interpretar un centroide como una
    intersección exacta.
    </div>
    """.replace(",", "."),
    unsafe_allow_html=True,
)

st.write("")
center = [float(mapped["latitud"].median()), float(mapped["longitud"].median())]
map_object = folium.Map(
    location=center,
    zoom_start=11 if municipalities else 9,
    tiles="OpenStreetMap",
    control_scale=True,
)

if show_heatmap:
    heat_group = folium.FeatureGroup(name="Densidad general", show=True)
    HeatMap(
        mapped[["latitud", "longitud"]].astype(float).values.tolist(),
        radius=16,
        blur=13,
        min_opacity=0.3,
        gradient={0.25: "#75B6F1", 0.55: "#2879C8", 0.8: "#E97824", 1: "#C54B4B"},
    ).add_to(heat_group)
    heat_group.add_to(map_object)

if show_markers:
    marker_group = folium.FeatureGroup(name="Registros", show=True)
    marker_cluster = MarkerCluster().add_to(marker_group)
    for row in mapped.itertuples(index=False):
        event_date = (
            row.fecha_hecho.strftime("%Y-%m-%d")
            if pd.notna(row.fecha_hecho)
            else "Sin fecha"
        )
        event_class = (
            row.clase_siniestro_modelo
            if pd.notna(row.clase_siniestro_modelo)
            else "No modelable"
        )
        popup = (
            f"<strong>{escape(str(row.municipio_hecho))}</strong><br>"
            f"Fecha: {escape(event_date)}<br>"
            f"Vehículo: {escape(str(row.vehiculo))}<br>"
            f"Clase: {escape(str(event_class))}"
        )
        folium.Marker(
            location=[float(row.latitud), float(row.longitud)],
            popup=folium.Popup(popup, max_width=280),
            tooltip=str(event_class),
        ).add_to(marker_cluster)
    marker_group.add_to(map_object)

if show_hotspots and not hotspots.empty:
    hotspot_group = folium.FeatureGroup(name="Concentraciones candidatas", show=True)
    for hotspot in hotspots.itertuples(index=False):
        popup = (
            f"<strong>Candidato {hotspot.prioridad_visual}</strong><br>"
            f"{escape(str(hotspot.municipio))} · {escape(str(hotspot.sector))}<br>"
            f"Registros: {hotspot.registros}<br>"
            f"Clase predominante: {escape(str(hotspot.clase_predominante))}<br>"
            f"Participación: {hotspot.participacion_clase:.1%}"
        )
        folium.Circle(
            location=[float(hotspot.latitud), float(hotspot.longitud)],
            radius=float(radius_m),
            color="#C54B4B",
            weight=2,
            fill=True,
            fill_color="#E97824",
            fill_opacity=0.2,
            tooltip=f"Candidato {hotspot.prioridad_visual}: {hotspot.registros} registros",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(hotspot_group)
        folium.CircleMarker(
            location=[float(hotspot.latitud), float(hotspot.longitud)],
            radius=6,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color="#C54B4B",
            fill_opacity=1,
        ).add_to(hotspot_group)
    hotspot_group.add_to(map_object)

folium.LayerControl(collapsed=False).add_to(map_object)
st_folium(map_object, use_container_width=True, height=620, returned_objects=[])
st.caption(
    "El mapa de calor usa todas las coordenadas dentro de rangos válidos. Las áreas resaltadas "
    "son candidatas para revisión, no una clasificación oficial de puntos críticos."
)

section_header(
    "De hallazgo a acción",
    "Recomendación explicable por concentración",
    "Seleccione un candidato y, si conoce la geometría del lugar, complete su contexto.",
)

if hotspots.empty:
    st.info(
        "No se detectaron concentraciones con los parámetros actuales. Pruebe un radio mayor, "
        "un mínimo menor o una combinación de filtros menos restrictiva."
    )
else:
    selected_label = st.selectbox(
        "Concentración candidata",
        hotspots["etiqueta"].tolist(),
    )
    selected = hotspots.loc[hotspots["etiqueta"].eq(selected_label)].iloc[0]
    cluster_group = clustered.loc[clustered["cluster_id"].eq(selected["cluster_id"])].copy()

    contexts = ["No especificado", "Intersección", "Rotonda o glorieta", "Tramo vial"]
    detected_context = str(selected["contexto_detectado"])
    context_index = contexts.index(detected_context) if detected_context in contexts else 0
    site_context = st.selectbox(
        "Contexto del sitio",
        contexts,
        index=context_index,
        help="El texto de la dirección puede sugerir un contexto; confírmelo con conocimiento local o visita de campo.",
    )

    cards = st.columns(4)
    cards[0].metric("Registros cercanos", int(selected["registros"]))
    cards[1].metric("Clase predominante", str(selected["clase_predominante"]).title())
    cards[2].metric("Participación de la clase", f"{selected['participacion_clase']:.1%}")
    cards[3].metric("Asociados a motocicleta", f"{selected['participacion_moto']:.1%}")

    left, right = st.columns([1, 1.15])
    with left:
        with st.container(border=True):
            class_distribution = (
                cluster_group["clase_siniestro_modelo"]
                .fillna("NO MODELABLE")
                .value_counts()
                .rename_axis("clase")
                .reset_index(name="registros")
                .sort_values("registros")
            )
            fig = px.bar(
                class_distribution,
                x="registros",
                y="clase",
                orientation="h",
                title="Composición del candidato",
                color="registros",
                color_continuous_scale=["#CFE5F8", COLORS["blue"]],
                text_auto=True,
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(title="Registros")
            fig.update_yaxes(title="")
            st.plotly_chart(style_chart(fig, 360), width="stretch")

    with right:
        recommendation = hotspot_recommendation(cluster_group, site_context)
        with st.container(border=True):
            st.markdown("#### Frente recomendado para evaluación")
            st.markdown(f"**{recommendation['focus']}**")
            for action in recommendation["actions"]:
                st.markdown(f"- {action}")
            st.warning(recommendation["disclaimer"])

    date_start = selected["fecha_inicial"]
    date_end = selected["fecha_final"]
    if pd.notna(date_start) and pd.notna(date_end):
        st.caption(
            f"Periodo observado en este candidato: {date_start:%Y-%m-%d} a {date_end:%Y-%m-%d}. "
            "La repetición histórica orienta la inspección, pero no mide exposición al tránsito."
        )

render_footer()
