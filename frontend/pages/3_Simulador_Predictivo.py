from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.api_client import ApiUnavailable, model_options, predict
from src.data import historical_context, load_data
from src.ui import COLORS, configure_page, page_header, render_footer, style_chart


configure_page("Simulador predictivo", "🧠")
page_header(
    "Módulo 3 · apoyo predictivo",
    "Simulador de escenarios preventivos",
    "Estime la clase de siniestro asociada a un escenario, contrástela con casos históricos y defina qué frente preventivo revisar.",
    badge="Random Forest · resultado experimental y explicable",
)

st.info(
    "**Pregunta que responde este módulo:** si ocurriera un siniestro fatal bajo las "
    "características ingresadas, ¿qué clase sería más plausible, cómo se compara con "
    "registros semejantes y qué frente preventivo conviene revisar?"
)

st.markdown(
    "El **Dashboard** muestra cuándo y en qué grupos aparecen los patrones; el **Mapa** "
    "muestra dónde se concentran; este simulador propone **qué tipo de evento y qué "
    "medidas revisar**. La priorización se sustenta con los tres módulos."
)

st.markdown(
    """
    <div class="experimental">
      <strong>Modelo experimental:</strong> F1-Macro OOF = 0,631 frente a la meta 0,70.
      El modelo clasifica siniestros ya ocurridos; no calcula la probabilidad de que ocurra
      un siniestro ni demuestra causalidad.
    </div>
    """,
    unsafe_allow_html=True,
)

options = model_options()
categories = options["categories"]
data = load_data()


def select_category(label: str, feature: str, key: str) -> str:
    values = categories[feature]
    preferred = {
        "municipio_hecho": "PEREIRA",
        "sexo": "MASCULINO",
        "zona_hecho": "URBANA",
        "vehiculo": "MOTOCICLETA",
    }.get(feature)
    index = values.index(preferred) if preferred in values else 0
    return st.selectbox(label, values, index=index, key=key)


with st.form("prediction_form"):
    st.subheader("Defina el escenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Actor vial**")
        edad = st.number_input("Edad", min_value=0, max_value=110, value=35)
        sexo = select_category("Sexo", "sexo", "sexo")
        condicion_victima = select_category(
            "Condición de la víctima", "condicion_victima", "condicion_victima"
        )
        vehiculo = select_category("Vehículo", "vehiculo", "vehiculo")
    with col2:
        st.markdown("**Territorio y tiempo**")
        municipio = select_category("Municipio", "municipio_hecho", "municipio")
        zona = select_category("Zona", "zona_hecho", "zona")
        dia = select_category("Día de la semana", "dia_semana", "dia")
        mes = st.number_input("Mes", min_value=1, max_value=12, value=8)
        hora_desconocida = st.checkbox("Hora no disponible")
        hora = st.number_input(
            "Hora decimal",
            min_value=0.0,
            max_value=23.99,
            value=18.0,
            step=0.25,
            disabled=hora_desconocida,
        )
    with col3:
        st.markdown("**Entorno vial**")
        escenario = select_category("Escenario", "escenario_hecho", "escenario")
        condicion_lugar = select_category(
            "Condición del lugar", "condicion_lugar", "condicion_lugar"
        )
        estado_via = select_category("Estado de la vía", "estado_via", "estado_via")
        tipo_servicio = select_category(
            "Tipo de servicio", "tipo_servicio_vehiculo", "tipo_servicio"
        )

    submitted = st.form_submit_button(
        "Analizar escenario", type="primary", width="stretch"
    )

if submitted:
    payload = {
        "edad": edad,
        "hora": None if hora_desconocida else hora,
        "sexo": sexo,
        "zona_hecho": zona,
        "vehiculo": vehiculo,
        "condicion_victima": condicion_victima,
        "dia_semana": dia,
        "municipio_hecho": municipio,
        "mes_hecho": mes,
        "tipo_servicio_vehiculo": tipo_servicio,
        "escenario_hecho": escenario,
        "condicion_lugar": condicion_lugar,
        "estado_via": estado_via,
    }
    try:
        result = predict(payload)
    except (ApiUnavailable, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["simulator_payload"] = payload
    st.session_state["simulator_result"] = result

result = st.session_state.get("simulator_result")
payload = st.session_state.get("simulator_payload")

if result and payload:
    history = historical_context(data, payload, result["prediction"])
    decision = result["decision_support"]

    st.divider()
    st.subheader("Respuesta del escenario")
    st.write(decision["answer"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clase más plausible", result["prediction"])
    col2.metric("Puntaje del modelo", f"{100 * result['confidence']:.1f}%")
    col3.metric(
        "Casos históricos comparables",
        f"{history['records']:,}".replace(",", "."),
    )
    agreement = "Sí" if history["aligned"] else "No"
    col4.metric("Coincide con la clase histórica principal", agreement)

    st.caption(
        f"Señal predictiva {decision['signal_strength'].lower()}: "
        f"{decision['signal_definition']}"
    )

    if history["aligned"]:
        st.success(
            f"El modelo y los casos históricos comparables coinciden en **{result['prediction']}**. "
            "Esto respalda revisar el frente preventivo propuesto, sujeto a validación territorial."
        )
    else:
        st.warning(
            f"El modelo estima **{result['prediction']}**, pero la clase más frecuente entre los "
            f"casos comparables es **{history['top_class']}**. Revise ambas señales antes de priorizar."
        )

    st.subheader("Evidencia del resultado")
    model_distribution = pd.DataFrame(
        {
            "clase": list(result["probabilities"].keys()),
            "Modelo": list(result["probabilities"].values()),
        }
    )
    comparison = model_distribution.merge(
        history["distribution"][["clase", "participacion_historica"]],
        on="clase",
        how="outer",
    ).fillna(0.0)
    comparison = comparison.rename(
        columns={"participacion_historica": "Histórico comparable"}
    )
    chart_data = comparison.melt(
        id_vars="clase",
        value_vars=["Modelo", "Histórico comparable"],
        var_name="fuente",
        value_name="participacion",
    )
    fig = px.bar(
        chart_data,
        x="participacion",
        y="clase",
        color="fuente",
        barmode="group",
        orientation="h",
        text_auto=".1%",
        color_discrete_map={
            "Modelo": COLORS["navy"],
            "Histórico comparable": COLORS["orange"],
        },
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        xaxis_title="Participación dentro de cada fuente",
        yaxis_title="",
        legend_title="",
    )
    st.plotly_chart(style_chart(fig, 410), width="stretch")
    st.caption(
        f"Comparación histórica basada en {history['records']} registros. Criterios usados: "
        f"{history['scope']}. La participación histórica describe siniestros registrados; "
        "no es una tasa de ocurrencia."
    )

    with st.expander("Ver la lectura técnica del modelo"):
        st.write(
            f"La segunda clase es **{decision['runner_up_class']}** con "
            f"{decision['runner_up_probability']:.1%}. La diferencia entre las dos primeras "
            f"clases es {decision['probability_margin']:.1%}."
        )
        st.write(
            f"La clase estimada representa {decision['historical_baseline']:.1%} del histórico "
            f"de entrenamiento. El puntaje del escenario equivale a "
            f"{result['relative_index']:.2f} veces esa participación."
        )
        st.caption(result["relative_index_definition"])

    st.subheader("Qué decisión apoya")
    recommendation = result["recommendation"]
    st.markdown(f"#### {recommendation['title']}")
    st.write(recommendation["rationale"])
    for action in recommendation["actions"]:
        st.markdown(f"- {action}")

    st.markdown("**Validación antes de priorizar una intervención**")
    for step, validation in enumerate(recommendation["validation_steps"], start=1):
        st.markdown(f"{step}. {validation}")

    nav1, nav2 = st.columns(2)
    if nav1.button("Validar contexto en el Dashboard", width="stretch"):
        st.session_state["analysis_context"] = {
            "municipalities": [payload["municipio_hecho"]],
            "zones": [payload["zona_hecho"]],
            "vehicles": [payload["vehiculo"]],
        }
        st.switch_page("pages/1_Dashboard_EDA.py")
    if nav2.button("Localizar concentraciones en el Mapa", width="stretch"):
        st.session_state["analysis_context"] = {
            "municipalities": [payload["municipio_hecho"]],
            "zones": [payload["zona_hecho"]],
            "vehicles": [payload["vehiculo"]],
        }
        st.switch_page("pages/2_Mapa_Geoespacial.py")

    for warning in result["warnings"]:
        st.warning(warning)

    with st.expander("Alcance y limitaciones"):
        st.markdown(
            "- No estima la probabilidad de que ocurra un siniestro.\n"
            "- No estima severidad, número de víctimas ni efecto causal de una variable.\n"
            "- No reemplaza la inspección del sitio ni el criterio de la autoridad.\n"
            "- El desempeño global está por debajo de la meta F1-Macro de 0,70."
        )
        st.caption(recommendation["disclaimer"])

render_footer()
