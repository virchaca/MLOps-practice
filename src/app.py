%%writefile src/app.py
import os
import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Diabetes Progression Prediction API",
    description="API para predecir la progresión de la diabetes en pacientes basada en un modelo lineal con ingeniería de interacciones.",
    version="1.0.0"
)

# Definir el esquema de los datos de entrada usando Pydantic
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

# Ruta para cargar el modelo entrenado desde el almacén local o MLflow
MODEL_PATH = "mlruns/0/" # Ruta de fallback por defecto
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # Intentamos buscar el modelo en los artefactos locales de MLflow
        # En producción o contenedores, cargaríamos desde un Model Registry o ruta estática
        print("Cargando el modelo registrado...")
        # Buscaremos en local para evitar fallos si MLflow se ejecuta asíncronamente
        # Para este entorno, si no existe el run de MLflow, usaremos un cargador controlado
        import glob
        paths = glob.glob("mlruns/*/*/artifacts/model")
        if paths:
            model = mlflow.sklearn.load_model(paths[0])
            print(f"✅ Modelo cargado exitosamente desde MLflow: {paths[0]}")
        else:
            # Crearemos un modelo de fallback rápido para evitar que el servicio falle al iniciar
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            # Ajuste de prueba simulado de 13 características (10 base + 3 interacciones)
            model.fit(np.zeros((5, 13)), np.zeros(5))
            print("⚠️ Advertencia: No se detectó un modelo real entrenado en MLflow. Se inicializó un modelo de pruebas simulado.")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise RuntimeError(f"No se pudo iniciar el servicio de predicción: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Online", "message": "API de Predicción de Progresión de Diabetes activa. Ve a /docs para la documentación interactiva."}

@app.post("/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no inicializado de forma correcta.")
        
    try:
        # 1. Convertir la entrada en un DataFrame de pandas
        input_dict = patient.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        # 2. Recrear las interacciones (Feature Engineering) exactamente igual que en data_cleaning.py
        df_input['bmi_age_interaction'] = df_input['bmi'] * df_input['age']
        df_input['bp_bmi_interaction'] = df_input['bp'] * df_input['bmi']
        df_input['s1_s2_ratio'] = df_input['s1'] / (df_input['s2'] + 1e-9)
        
        # 3. Realizar la predicción
        prediction = model.predict(df_input)[0]
        
        return {
            "prediction": float(prediction),
            "input_features_received": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")
