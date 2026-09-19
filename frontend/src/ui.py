from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st


COLORS = {
    "navy": "#123A63",
    "navy_dark": "#0A2947",
    "blue": "#2879C8",
    "blue_light": "#75B6F1",
    "teal": "#148A80",
    "green": "#1D8A72",
    "orange": "#E97824",
    "red": "#C54B4B",
    "background": "#F3F6FA",
    "muted": "#667085",
}


def configure_page(title: str, icon: str = "🚦") -> None:
    st.set_page_config(
        page_title=f"{title} | Siniestralidad vial",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --navy: #123A63;
            --navy-dark: #0A2947;
            --blue: #2879C8;
            --sky: #75B6F1;
            --border: #DDE5EE;
            --muted: #667085;
        }
        .stApp {
            background:
                radial-gradient(circle at 90% 0%, rgba(117,182,241,.12), transparent 28rem),
                #F3F6FA;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A2947 0%, #123A63 100%);
            border-right: 1px solid rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] * {
            color: #101828 !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #FFFFFF 0%, #FAFCFF 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 18px 16px;
            min-height: 126px;
            box-shadow: 0 7px 20px rgba(18,58,99,.07);
        }
        [data-testid="stMetricLabel"] { color: #475467; font-weight: 650; }
        [data-testid="stMetricValue"] { color: var(--navy-dark); font-weight: 800; }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.97);
            border-color: var(--border);
            border-radius: 16px;
            box-shadow: 0 7px 20px rgba(18,58,99,.06);
        }
        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 86% 20%, rgba(117,182,241,.32), transparent 24rem),
                linear-gradient(120deg, #0A2947 0%, #174E7C 62%, #1E659B 100%);
            color: #FFFFFF;
            border-radius: 20px;
            padding: 30px 34px;
            margin-bottom: 22px;
            box-shadow: 0 14px 36px rgba(10,41,71,.18);
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            border: 1px solid rgba(255,255,255,.14);
            border-radius: 50%;
            right: -80px;
            top: -130px;
        }
        .hero-eyebrow {
            color: #A9D8FF;
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }
        .hero h1 {
            color: #FFFFFF;
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1.04;
            margin: 0 0 .5rem 0;
            max-width: 900px;
        }
        .hero p {
            color: rgba(255,255,255,.84);
            font-size: 1.02rem;
            max-width: 920px;
            margin: 0;
        }
        .hero-badge {
            display: inline-block;
            margin-top: 16px;
            padding: 7px 11px;
            border-radius: 999px;
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.16);
            color: #FFFFFF;
            font-size: .82rem;
            font-weight: 650;
        }
        .module-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 17px;
            padding: 21px;
            min-height: 190px;
            box-shadow: 0 8px 24px rgba(18,58,99,.07);
        }
        .module-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: #E7F2FC;
            color: var(--blue);
            font-weight: 800;
            margin-bottom: 14px;
        }
        .module-card h3 { color: var(--navy-dark); margin: 0 0 8px 0; }
        .module-card p { color: var(--muted); margin-bottom: 0; }
        .experimental,
        .method-note {
            border-left: 5px solid #E97824;
            background: #FFF7ED;
            padding: 13px 15px;
            border-radius: 10px;
            color: #7A3811;
        }
        .method-note {
            border-left-color: #2879C8;
            background: #EEF6FE;
            color: #123A63;
        }
        .verdict-card {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 5px 16px rgba(18,58,99,.05);
        }
        .verdict-card h4 { color: var(--navy-dark); margin: 8px 0 7px; }
        .verdict-card p { margin: 5px 0; color: #475467; }
        .verdict-pill {
            display: inline-block;
            padding: 5px 9px;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .04em;
        }
        .verdict-supported { background: #E6F5F0; color: #166B59; }
        .verdict-refuted { background: #FDECEC; color: #A53636; }
        .verdict-unknown { background: #FFF3DD; color: #8B5713; }
        .verdict-metric { color: var(--navy-dark); font-size: 1.18rem; font-weight: 800; }
        .small-muted { color: #667085; font-size: .84rem; }
        .section-kicker {
            color: var(--blue);
            font-weight: 800;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 2px;
        }
        .section-title { color: var(--navy-dark); margin: 0 0 6px; }
        [data-baseweb="tab-list"] { gap: 8px; }
        [data-baseweb="tab"] {
            background: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: 10px 10px 0 0;
            padding-left: 18px;
            padding-right: 18px;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background: #EAF4FD;
            color: var(--navy-dark);
        }
        .stButton > button, .stLinkButton > a {
            border-radius: 10px;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    eyebrow: str,
    title: str,
    description: str,
    badge: str | None = None,
) -> None:
    badge_html = f'<div class="hero-badge">{escape(badge)}</div>' if badge else ""
    st.markdown(
        (
            '<section class="hero">'
            f'<div class="hero-eyebrow">{escape(eyebrow)}</div>'
            f'<h1>{escape(title)}</h1>'
            f'<p>{escape(description)}</p>'
            f"{badge_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str | None = None) -> None:
    description_html = (
        f'<div class="small-muted">{escape(description)}</div>' if description else ""
    )
    st.markdown(
        (
            f'<div class="section-kicker">{escape(kicker)}</div>'
            f'<h2 class="section-title">{escape(title)}</h2>'
            f"{description_html}"
        ),
        unsafe_allow_html=True,
    )


def style_chart(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#344054"},
        title={"font": {"size": 17, "color": COLORS["navy_dark"]}, "x": 0.01},
        margin={"l": 18, "r": 18, "t": 58, "b": 24},
        hoverlabel={"bgcolor": "#FFFFFF", "font_color": "#101828"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="#E8EEF5", zeroline=False)
    fig.update_yaxes(gridcolor="#E8EEF5", zeroline=False)
    return fig


def render_verdict_card(item: dict) -> None:
    verdict = str(item["verdict"])
    css_class = {
        "RESPALDADO": "verdict-supported",
        "DESMENTIDO": "verdict-refuted",
        "NO DEMOSTRABLE": "verdict-unknown",
    }.get(verdict, "verdict-unknown")
    st.markdown(
        (
            '<article class="verdict-card">'
            f'<span class="verdict-pill {css_class}">{escape(verdict)}</span>'
            f'<h4>{escape(str(item["claim"]))}</h4>'
            f'<div class="verdict-metric">{escape(str(item["metric"]))}</div>'
            f'<p>{escape(str(item["evidence"]))}</p>'
            f'<p class="small-muted"><strong>Alcance:</strong> {escape(str(item["limitation"]))}</p>'
            "</article>"
        ),
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.divider()
    st.caption(
        "Fuente: consolidado de siniestralidad vial fatal de Pereira y Risaralda · "
        "Prototipo académico de apoyo analítico · Las recomendaciones requieren validación técnica."
    )
