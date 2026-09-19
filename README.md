# Prototipo funcional de siniestralidad vial

Producto académico compuesto por una aplicación Streamlit y una API FastAPI. Conserva el alcance del prototipo fachada:

1. Dashboard EDA con resumen ejecutivo, análisis detallado y sección de mitos.
2. Mapa geoespacial con densidad, concentraciones candidatas y recomendaciones explicables.
3. Simulador de escenarios preventivos con Random Forest, contraste histórico y recomendaciones orientativas.

## Arquitectura

- `frontend/`: interfaz Streamlit y los tres módulos.
- `backend/`: API FastAPI que carga el artefacto predictivo.
- `artifacts/`: Pipeline completo y metadatos versionados.
- `data/`: consolidado utilizado por Dashboard y Mapa.
- `scripts/train_and_package.py`: entrenamiento final y serialización reproducible.

El Dashboard y el Mapa leen el consolidado directamente. El Simulador llama a `POST /predict`
y compara la respuesta con casos históricos semejantes. Desde el resultado se puede abrir el
Dashboard o el Mapa con municipio, zona y vehículo precargados.

## Qué responde cada módulo

- **Dashboard:** cuándo, en qué municipios y bajo qué características aparecen los registros. Incluye gráficas por año, clase, mes, franja horaria observada, condición de la víctima, edad y estado de la vía.
- **Mitos vs. datos:** contrasta afirmaciones como la mayoría motociclista, el predominio del fin de semana o la influencia de la edad. Separa hallazgos descriptivos de afirmaciones no demostrables por falta de población expuesta.
- **Mapa:** localiza concentraciones candidatas y sugiere qué revisar según la clase predominante y el contexto del sitio. En una glorieta puede proponer evaluar semaforización, pero nunca prescribe su instalación sin estudio operacional y visita de campo.
- **Simulador:** si se plantea un siniestro fatal hipotético, estima qué clase resulta más plausible y qué frente preventivo conviene contrastar con el Dashboard y el Mapa.

No se muestran indicadores de heridos, severidad o totales de accidentes no fatales porque las fuentes consolidadas no contienen esas variables con cobertura suficiente.

## Calidad espacial y temporal

- Una coordenada dentro de rangos válidos no necesariamente tiene precisión de intersección.
- Para recomendaciones, se excluyen coordenadas repetidas que representan múltiples direcciones y parecen corresponder a ubicaciones agregadas.
- Las concentraciones se detectan por proximidad, con radio y número mínimo configurables; son candidatas a inspección, no puntos críticos oficiales.
- Las visualizaciones por franja horaria excluyen las horas exactamente 00:00, debido a su uso repetido como valor técnico en las fuentes.

## Resultado predictivo vigente

- Modelo: Random Forest.
- Evaluación: predicciones OOF con 4 folds estratificados y agrupados.
- F1-Macro OOF: 0,631.
- Meta D1: 0,70, todavía no cumplida.
- Uso permitido: demostración experimental y apoyo exploratorio.

## Ejecución local en macOS

Desde la raíz del proyecto:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Si `python3.13` no existe, use un Python 3.11 o superior. Después de activar el entorno,
`which python` debe terminar en `producto_siniestralidad/.venv/bin/python`.

Inicie la API en una terminal:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

La documentación interactiva quedará en `http://localhost:8000/docs`.

En otra terminal, inicie Streamlit:

```bash
source .venv/bin/activate
python -m streamlit run frontend/app.py --global.developmentMode=false
```

La aplicación quedará en `http://localhost:8501`.

## Reentrenar y empaquetar

```bash
python scripts/train_and_package.py \
  --data data/Datos_consolidados_siniestralidad_vial.xlsx \
  --output-dir artifacts
```

El script serializa el preprocesamiento y el Random Forest como un único Pipeline. El artefacto no depende de transformaciones manuales realizadas por la interfaz.

## Pruebas

```powershell
pytest
```

Las pruebas verifican carga del modelo, contrato de predicción, probabilidades, validación numérica, manejo de categorías nuevas, control de calidad espacial, detección de concentraciones y reglas de recomendaciones.

## Docker

Con Docker Desktop:

```powershell
docker compose up --build
```

Servicios:

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

## Variables de entorno

Use `.env.example` como referencia. En producción limite `CORS_ALLOWED_ORIGINS` al dominio real del frontend. No se usan credenciales en el navegador para consumir la API.

## Contrato principal

`POST /predict` recibe las 13 variables del conjunto Contextual y devuelve:

- clase estimada;
- confianza y probabilidades por clase;
- índice relativo frente a la prevalencia histórica;
- recomendación orientativa;
- advertencias de incertidumbre;
- versión y contexto de métricas del modelo.

El índice relativo no es causal ni equivale a una estimación de severidad o de ocurrencia.
El simulador responde qué clase sería más plausible si ocurriera un siniestro fatal bajo el
escenario ingresado. La decisión preventiva se valida con patrones del Dashboard y
concentraciones del Mapa.
