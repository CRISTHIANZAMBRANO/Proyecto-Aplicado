from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    edad: float = Field(ge=0, le=110, examples=[35])
    hora: float | None = Field(default=None, ge=0, lt=24, examples=[18.5])
    sexo: str = Field(min_length=1, examples=["MASCULINO"])
    zona_hecho: str = Field(min_length=1, examples=["URBANA"])
    vehiculo: str = Field(min_length=1, examples=["MOTOCICLETA"])
    condicion_victima: str = Field(min_length=1, examples=["CONDUCTOR"])
    dia_semana: str = Field(min_length=1, examples=["VIERNES"])
    municipio_hecho: str = Field(min_length=1, examples=["PEREIRA"])
    mes_hecho: int | None = Field(default=None, ge=1, le=12, examples=[8])
    tipo_servicio_vehiculo: str = Field(min_length=1, examples=["PARTICULAR"])
    escenario_hecho: str = Field(min_length=1, examples=["VIA PUBLICA"])
    condicion_lugar: str = Field(min_length=1, examples=["SECO"])
    estado_via: str = Field(min_length=1, examples=["BUENO"])

    @field_validator(
        "sexo",
        "zona_hecho",
        "vehiculo",
        "condicion_victima",
        "dia_semana",
        "municipio_hecho",
        "tipo_servicio_vehiculo",
        "escenario_hecho",
        "condicion_lugar",
        "estado_via",
    )
    @classmethod
    def normalize_category(cls, value: str) -> str:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            raise ValueError("La categoría no puede estar vacía.")
        return cleaned.upper()


class Recommendation(BaseModel):
    title: str
    rationale: str
    actions: list[str]
    validation_steps: list[str]
    disclaimer: str


class DecisionSupport(BaseModel):
    question_answered: str
    answer: str
    signal_strength: str
    signal_definition: str
    historical_baseline: float
    runner_up_class: str
    runner_up_probability: float
    probability_margin: float


class MetricContext(BaseModel):
    f1_macro_oof: float
    target_f1_macro: float
    target_met: bool


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    confidence_level: str
    probabilities: dict[str, float]
    relative_index: float
    relative_index_definition: str
    recommendation: Recommendation
    decision_support: DecisionSupport
    warnings: list[str]
    model_version: str
    experimental: bool
    metric_context: MetricContext


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
