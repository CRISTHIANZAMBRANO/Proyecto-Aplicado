from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from app.services.model_service import ModelService, get_model_service
from app.settings import API_TITLE, API_VERSION, allowed_origins


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model_service()
    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "API experimental de apoyo a decisiones preventivas. Clasifica la clase de siniestro "
        "asociada a un escenario mediante el Pipeline Random Forest seleccionado y devuelve "
        "su interpretación y frente preventivo. Documentación interactiva en /docs."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/", tags=["Servicio"])
def root() -> dict:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Servicio"])
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=service.version,
    )


@app.get("/model/metadata", tags=["Modelo"])
def metadata(service: ModelService = Depends(get_model_service)) -> dict:
    return service.metadata


@app.get("/model/options", tags=["Modelo"])
def options(service: ModelService = Depends(get_model_service)) -> dict:
    return service.options()


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(
    payload: PredictionRequest,
    service: ModelService = Depends(get_model_service),
) -> PredictionResponse:
    return PredictionResponse.model_validate(service.predict(payload))
