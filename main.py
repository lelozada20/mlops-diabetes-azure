
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="API MLOps Diabetes",
    description="API conectada al modelo de diabetes de Databricks",
    version="1.0.0",
)

DATABRICKS_URL = os.getenv("DATABRICKS_URL", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Glucose_BMI_interaction",
    "Age_40_plus",
]


class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0)
    Glucose: float = Field(ge=0)
    BloodPressure: float = Field(ge=0)
    SkinThickness: float = Field(ge=0)
    Insulin: float = Field(ge=0)
    BMI: float = Field(ge=0)
    DiabetesPedigreeFunction: float = Field(ge=0)
    Age: int = Field(ge=0)


@app.get("/")
def inicio():
    return {"mensaje": "API MLOps Diabetes activa"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {
        "nombre": "API MLOps Diabetes",
        "modelo": "workspace.default.mlops_diabetes_model",
        "endpoint": "diabetes-serving",
    }


@app.post("/predecir")
def predecir(datos: DiabetesInput):
    if not DATABRICKS_URL or not DATABRICKS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Faltan las variables DATABRICKS_URL o DATABRICKS_TOKEN",
        )

    entrada = datos.model_dump()
    entrada["Glucose_BMI_interaction"] = (
        entrada["Glucose"] / 100.0
    ) * (
        entrada["BMI"] / 30.0
    )
    entrada["Age_40_plus"] = int(entrada["Age"] >= 40)

    payload = {
        "dataframe_split": {
            "columns": FEATURES,
            "data": [[entrada[col] for col in FEATURES]],
        }
    }

    try:
        respuesta = requests.post(
            DATABRICKS_URL,
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        respuesta.raise_for_status()
        resultado = respuesta.json()

        if "predictions" not in resultado:
            raise HTTPException(
                status_code=502,
                detail="Databricks no devolvió predicciones",
            )

        return {"predictions": resultado["predictions"]}

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="No fue posible consultar el modelo de Databricks",
        )


@app.get("/docs-info")
def docs_info():
    return {
        "documentacion": "/docs",
        "rutas": ["/", "/health", "/info", "/predecir", "/docs-info"],
    }
