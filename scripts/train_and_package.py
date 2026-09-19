from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


TARGET = "clase_siniestro_modelo"
GROUP = "grupo_evento"
RANDOM_STATE = 42
FEATURES = [
    "edad",
    "hora",
    "sexo",
    "zona_hecho",
    "vehiculo",
    "condicion_victima",
    "dia_semana",
    "municipio_hecho",
    "mes_hecho",
    "tipo_servicio_vehiculo",
    "escenario_hecho",
    "condicion_lugar",
    "estado_via",
]
NUMERIC_FEATURES = ["edad", "hora", "mes_hecho"]
CATEGORICAL_FEATURES = [column for column in FEATURES if column not in NUMERIC_FEATURES]

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 24,
    "min_samples_leaf": 8,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

VERIFIED_METRICS = {
    "f1_macro_oof": 0.6309969191402276,
    "balanced_accuracy_oof": 0.6518815118308638,
    "accuracy_oof": 0.742145178764897,
    "target_f1_macro": 0.70,
    "target_met": False,
    "evaluation": "4-fold StratifiedGroupKFold out-of-fold",
}


def make_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            (
                "imputation",
                SimpleImputer(strategy="median", add_indicator=True),
            )
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputation",
                SimpleImputer(strategy="constant", fill_value="SIN INFORMACION"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(**MODEL_PARAMS)
    return Pipeline(
        [
            ("preparation", preprocessor),
            ("model", classifier),
        ]
    )


def clean_category(value: object) -> str:
    if pd.isna(value):
        return "SIN INFORMACION"
    return str(value).strip()


def build_metadata(model_data: pd.DataFrame, encoder: LabelEncoder) -> dict:
    target_counts = model_data[TARGET].value_counts().reindex(encoder.classes_)
    priors = (target_counts / target_counts.sum()).to_dict()
    categories = {
        column: sorted({clean_category(value) for value in model_data[column].tolist()})
        for column in CATEGORICAL_FEATURES
    }
    return {
        "model_name": "Random Forest - clase de siniestro",
        "model_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experimental": True,
        "warning": (
            "Modelo experimental de apoyo analítico. No sustituye el criterio de una autoridad "
            "ni demuestra relaciones causales."
        ),
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categories": categories,
        "classes": encoder.classes_.tolist(),
        "class_priors": {str(key): float(value) for key, value in priors.items()},
        "training_rows": int(len(model_data)),
        "training_groups": int(model_data[GROUP].nunique()),
        "verified_metrics": VERIFIED_METRICS,
        "model_parameters": MODEL_PARAMS,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y empaqueta el Pipeline final.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.read_excel(args.data, sheet_name="Datos_consolidados")
    model_data = full.loc[full["objetivo_modelable"].eq(True)].copy()

    missing_columns = sorted(set(FEATURES + [TARGET, GROUP]) - set(model_data.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {missing_columns}")
    if len(model_data) != 1846:
        raise ValueError(f"Se esperaban 1.846 filas modelables y se encontraron {len(model_data)}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(model_data[TARGET])
    weights = compute_sample_weight(class_weight="balanced", y=y)
    pipeline = make_pipeline()
    pipeline.fit(model_data[FEATURES], y, model__sample_weight=weights)

    artifact = {
        "pipeline": pipeline,
        "label_encoder": encoder,
        "features": FEATURES,
    }
    model_path = args.output_dir / "model_pipeline.joblib"
    metadata_path = args.output_dir / "model_metadata.json"
    joblib.dump(artifact, model_path, compress=3)
    metadata = build_metadata(model_data, encoder)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reloaded = joblib.load(model_path)
    smoke_prediction = reloaded["pipeline"].predict(model_data[FEATURES].head(1))
    if smoke_prediction.shape != (1,):
        raise RuntimeError("El artefacto serializado no superó la prueba de recarga.")

    print(json.dumps({
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "training_rows": len(model_data),
        "classes": encoder.classes_.tolist(),
        "smoke_prediction": encoder.inverse_transform(smoke_prediction).tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
