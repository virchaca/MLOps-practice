from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# Mismo nombre y alias que asigna src/train.py
MODEL_NAME = "LinearRegressionDiabetesModel"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "bmi_age_interaction",
    "bp_bmi_interaction",
    "s1_s2_ratio",
]

model = None
model_metadata = None


class PatientData(BaseModel):
    age: float = Field(..., description="Edad (normalizada)")
    sex: float = Field(..., description="Sexo (normalizado)")
    bmi: float = Field(..., description="Índice de Masa Corporal (normalizado)")
    bp: float = Field(..., description="Presión sanguínea media (normalizada)")
    s1: float = Field(..., description="Medida de suero s1")
    s2: float = Field(..., description="Medida de suero s2")
    s3: float = Field(..., description="Medida de suero s3")
    s4: float = Field(..., description="Medida de suero s4")
    s5: float = Field(..., description="Medida de suero s5")
    s6: float = Field(..., description="Medida de suero s6")


def load_champion_model():
    """Carga el modelo con alias champion. Si no existe, el servicio sigue en marcha."""
    global model, model_metadata
    model = None
    model_metadata = None
    try:
        version_info = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(MODEL_URI)
        model_metadata = {
            "model_name": MODEL_NAME,
            "model_version": version_info.version,
            "alias": MODEL_ALIAS,
            "run_id": version_info.run_id,
            "status": version_info.status,
            "source": version_info.source,
        }
        print(f"Modelo champion cargado: {MODEL_NAME} versión {version_info.version}")
    except Exception as exc:
        print(f"Modelo champion no disponible: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_champion_model()
    yield


app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="API para predecir la progresión de la diabetes con el modelo champion de regresión lineal.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_model():
    if model is None or model_metadata is None:
        raise HTTPException(status_code=503, detail="Modelo champion no disponible.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    require_model()
    return model_metadata


@app.post("/predict")
def predict(patient: PatientData):
    require_model()
    try:
        row = patient.model_dump()
        row["bmi_age_interaction"] = row["bmi"] * row["age"]
        row["bp_bmi_interaction"] = row["bp"] * row["bmi"]
        row["s1_s2_ratio"] = row["s1"] / (row["s2"] + 1e-9)
        df_input = pd.DataFrame([row])[FEATURE_COLUMNS]
        prediction = model.predict(df_input)[0]
        return {
            "prediction": float(prediction),
            "model_name": model_metadata["model_name"],
            "model_version": model_metadata["model_version"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {exc}") from exc
