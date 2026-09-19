from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas import PredictionRequest
from app.settings import MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH


RECOMMENDATIONS = {
    "ATROPELLO": {
        "title": "Priorizar protección de actores viales vulnerables",
        "rationale": (
            "El escenario se asocia principalmente con atropello dentro de las clases "
            "aprendidas por el modelo. Esto orienta la revisión hacia la interacción entre "
            "peatones y vehículos."
        ),
        "actions": [
            "Revisar cruces peatonales, iluminación y señalización del sector.",
            "Considerar medidas de gestión de velocidad en la franja evaluada.",
            "Revisar continuidad de andenes, tiempos semafóricos y visibilidad en cruces.",
        ],
        "validation_steps": [
            "Ubicar en el mapa las concentraciones del municipio y la zona seleccionados.",
            "Confirmar en el Dashboard si el patrón se repite por día, vehículo y periodo.",
        ],
    },
    "CHOQUE CON OBJETO FIJO": {
        "title": "Revisar condiciones del corredor y elementos laterales",
        "rationale": (
            "El escenario se asocia principalmente con choque contra objeto fijo. La señal "
            "sirve para enfocar una inspección preventiva del corredor y su entorno lateral."
        ),
        "actions": [
            "Inspeccionar iluminación, demarcación y visibilidad del tramo.",
            "Revisar la ubicación y protección de elementos próximos a la vía.",
            "Revisar señales de velocidad, delineadores y condiciones de aproximación.",
        ],
        "validation_steps": [
            "Ubicar en el mapa los tramos con concentración de esta clase de siniestro.",
            "Confirmar en el Dashboard su relación descriptiva con horario y estado de vía.",
        ],
    },
    "CHOQUE CON OTRO VEHICULO": {
        "title": "Priorizar gestión de conflictos entre vehículos",
        "rationale": (
            "El escenario se asocia principalmente con choques entre vehículos. Esto orienta "
            "la revisión hacia intersecciones, maniobras y flujos conflictivos."
        ),
        "actions": [
            "Revisar intersecciones, maniobras y señalización del corredor.",
            "Evaluar medidas de control y separación de movimientos conflictivos.",
            "Revisar visibilidad, prioridades de paso y canalización de los flujos.",
        ],
        "validation_steps": [
            "Ubicar en el mapa concentraciones asociadas al municipio y zona evaluados.",
            "Confirmar en el Dashboard el patrón por franja temporal y tipo de vehículo.",
        ],
    },
    "OTROS": {
        "title": "Revisar el caso con información contextual adicional",
        "rationale": (
            "El escenario se concentra en la clase agrupada OTROS. Como reúne situaciones "
            "heterogéneas, requiere mayor validación antes de proponer una intervención."
        ),
        "actions": [
            "Consultar la distribución histórica de escenarios similares.",
            "Validar la calidad de los campos ingresados antes de actuar.",
            "Usar la predicción como señal exploratoria, no como decisión automática.",
        ],
        "validation_steps": [
            "Revisar en el Dashboard qué clases originales componen casos semejantes.",
            "Localizar en el mapa si existe una concentración territorial interpretable.",
        ],
    },
}


class ModelService:
    def __init__(self, artifact_path: Path, metadata_path: Path) -> None:
        if not artifact_path.exists():
            raise FileNotFoundError(f"No se encontró el artefacto: {artifact_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"No se encontraron metadatos: {metadata_path}")

        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.label_encoder = artifact["label_encoder"]
        self.features = artifact["features"]
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    def options(self) -> dict:
        return {
            "features": self.metadata["features"],
            "numeric_features": self.metadata["numeric_features"],
            "categorical_features": self.metadata["categorical_features"],
            "categories": self.metadata["categories"],
            "classes": self.metadata["classes"],
        }

    def predict(self, payload: PredictionRequest) -> dict:
        row = payload.model_dump()
        warnings = self._category_warnings(row)
        frame = pd.DataFrame([row], columns=self.features)

        encoded_prediction = self.pipeline.predict(frame)
        probabilities = self.pipeline.predict_proba(frame)[0]
        prediction = self.label_encoder.inverse_transform(encoded_prediction)[0]
        class_names = self.label_encoder.classes_.tolist()
        probability_map = {
            class_name: round(float(probability), 6)
            for class_name, probability in zip(class_names, probabilities)
        }

        confidence = float(np.max(probabilities))
        prior = float(self.metadata["class_priors"][prediction])
        relative_index = confidence / prior if prior else 0.0
        ranked_indices = np.argsort(probabilities)[::-1]
        runner_up_index = int(ranked_indices[1])
        runner_up_class = class_names[runner_up_index]
        runner_up_probability = float(probabilities[runner_up_index])
        probability_margin = confidence - runner_up_probability

        if confidence >= 0.70 and probability_margin >= 0.25:
            confidence_level = "ALTA"
        elif confidence >= 0.50 and probability_margin >= 0.12:
            confidence_level = "MEDIA"
        else:
            confidence_level = "BAJA"
            warnings.append(
                "La señal del modelo es poco diferenciada; revise la segunda clase y los casos históricos comparables."
            )

        recommendation = RECOMMENDATIONS[prediction]
        metrics = self.metadata["verified_metrics"]
        answer = (
            f"Si ocurriera un siniestro fatal bajo este escenario, el modelo asocia como "
            f"clase más plausible '{prediction}' ({confidence:.1%}), seguida de "
            f"'{runner_up_class}' ({runner_up_probability:.1%}). El resultado orienta el "
            f"frente preventivo '{recommendation['title'].lower()}', pero no estima la "
            "probabilidad de que ocurra un siniestro."
        )
        return {
            "prediction": prediction,
            "confidence": round(confidence, 6),
            "confidence_level": confidence_level,
            "probabilities": probability_map,
            "relative_index": round(relative_index, 3),
            "relative_index_definition": (
                "Puntaje del modelo dividido por la participación de la clase en el histórico "
                "de entrenamiento. No representa probabilidad de ocurrencia ni riesgo causal."
            ),
            "recommendation": {
                **recommendation,
                "disclaimer": self.metadata["warning"],
            },
            "decision_support": {
                "question_answered": (
                    "Si ocurriera un siniestro fatal bajo las características ingresadas, "
                    "¿qué clase sería más plausible y qué frente preventivo conviene revisar?"
                ),
                "answer": answer,
                "signal_strength": confidence_level,
                "signal_definition": (
                    "Grado de diferenciación entre las dos clases con mayor puntaje. "
                    "No equivale a exactitud del caso ni a probabilidad calibrada."
                ),
                "historical_baseline": round(prior, 6),
                "runner_up_class": runner_up_class,
                "runner_up_probability": round(runner_up_probability, 6),
                "probability_margin": round(probability_margin, 6),
            },
            "warnings": warnings,
            "model_version": self.version,
            "experimental": bool(self.metadata["experimental"]),
            "metric_context": {
                "f1_macro_oof": metrics["f1_macro_oof"],
                "target_f1_macro": metrics["target_f1_macro"],
                "target_met": metrics["target_met"],
            },
        }

    def _category_warnings(self, row: dict) -> list[str]:
        warnings = []
        for feature in self.metadata["categorical_features"]:
            value = row[feature]
            known = self.metadata["categories"][feature]
            if value not in known:
                warnings.append(
                    f"{feature}='{value}' no apareció en entrenamiento; se procesa como categoría desconocida."
                )
        return warnings


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService(MODEL_ARTIFACT_PATH, MODEL_METADATA_PATH)
