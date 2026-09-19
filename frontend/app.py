from __future__ import annotations

import streamlit as st

from src.api_client import ApiUnavailable, health
from src.data import load_data
from src.ui import configure_page, page_header, render_footer, section_header


configure_page("Inicio", "🚦")
page_header(
    "Plataforma de apoyo a decisiones",
    "Siniestralidad vial de Pereira y Risaralda",
    "Del patrón histórico a la acción por validar: explore cuándo ocurre, dónde se concentra y qué frente preventivo conviene estudiar.",
    badge="Datos descriptivos · análisis territorial · modelo experimental",
)

try:
    data = load_data()
    data_status = f"{len(data):,}".replace(",", ".")
    first_year = int(data["anio_hecho"].dropna().min())
    last_year = int(data["anio_hecho"].dropna().max())
    municipalities = int(data["municipio_hecho"].dropna().nunique())
    coordinate_coverage = float(data["coordenadas_validas"].eq(True).mean())
except Exception as exc:
    data_status = f"Datos no disponibles: {exc}"
    first_year = 0
    last_year = 0
    municipalities = 0
    coordinate_coverage = 0.0

try:
    api = health()
    api_status = f"API activa · modelo v{api['model_version']}"
    api_ok = True
except ApiUnavailable:
    api_status = "API predictiva desconectada"
    api_ok = False

metrics = st.columns(4)
metrics[0].metric(
    "Registros fatales",
    data_status,
    help="Una fila corresponde a una víctima fatal registrada en las fuentes consolidadas.",
)
metrics[1].metric("Cobertura temporal", f"{first_year}–{last_year}")
metrics[2].metric("Municipios representados", municipalities)
metrics[3].metric(
    "Coordenadas válidas",
    f"{coordinate_coverage:.1%}",
    help="Validez de rango; la precisión espacial se controla aparte en el mapa.",
)

st.write("")
section_header(
    "Ruta analítica",
    "Tres módulos, una decisión mejor sustentada",
    "Cada módulo responde una parte distinta de la pregunta del proyecto.",
)

columns = st.columns(3)
cards = [
    (
        "01",
        "Dashboard y mitos",
        "Identifica patrones temporales, territoriales y poblacionales; además contrasta afirmaciones populares con los datos.",
        "pages/1_Dashboard_EDA.py",
        "Explorar patrones",
    ),
    (
        "02",
        "Mapa y recomendaciones",
        "Detecta concentraciones candidatas y propone revisiones de seguridad vial mediante reglas transparentes.",
        "pages/2_Mapa_Geoespacial.py",
        "Priorizar zonas",
    ),
    (
        "03",
        "Simulador predictivo",
        "Clasifica un escenario hipotético, lo contrasta con casos semejantes y orienta el frente que conviene revisar.",
        "pages/3_Simulador_Predictivo.py",
        "Simular escenario",
    ),
]
for column, (number, title, description, page, action) in zip(columns, cards):
    with column:
        st.markdown(
            (
                '<div class="module-card">'
                f'<div class="module-number">{number}</div>'
                f'<h3>{title}</h3><p>{description}</p>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        st.page_link(page, label=action, width="stretch")

if not api_ok:
    st.info(
        "El Dashboard y el Mapa pueden utilizarse sin la API. Para activar el Simulador, "
        "inicie FastAPI en el puerto 8000."
    )
else:
    st.success(api_status)

left, right = st.columns([1.3, 1])
with left:
    with st.container(border=True):
        st.markdown("### Qué valor agrega")
        st.markdown(
            "La plataforma no se limita a mostrar gráficas: conecta una concentración "
            "geográfica con los tipos de evento observados y genera una lista explicable de "
            "acciones que la autoridad o el equipo técnico debería validar en campo."
        )
        st.markdown(
            "**Ejemplo:** si una glorieta concentra choques entre vehículos, el sistema "
            "sugiere revisar velocidades de entrada, prioridades, carriles y la viabilidad de "
            "controles como semaforización. No afirma que una medida deba instalarse sin estudio."
        )

with right:
    st.markdown(
        """
        <div class="experimental">
          <strong>Estado del modelo predictivo:</strong><br>
          candidato experimental con F1-Macro OOF = 0,631 frente a la meta 0,70.
          Su resultado se contrasta con el histórico y no se presenta como una certeza.
        </div>
        """,
        unsafe_allow_html=True,
    )
render_footer()
