
%%writefile src/train.py
import pandas as pd
import numpy as np
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(data_path="data/clean_data.csv", model_dir="models"):
    """Entrena el modelo de regresión lineal con MLflow tracking."""
    print(f"Iniciando el entrenamiento con el dataset: {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Error: El archivo {data_path} no existe. Por favor ejecuta la limpieza primero.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Separar características y variable objetivo
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Partición de datos entrenar/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear directorio para almacenar modelos de manera local si es necesario
    os.makedirs(model_dir, exist_ok=True)
    
    # Iniciar el tracking con MLflow
    mlflow.set_experiment("Diabetes_Progression_Experiment")
    
    with mlflow.start_run():
        # Definir el modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predicciones
        predictions = model.predict(X_test)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Métricas del modelo entrenado:")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAE:  {mae:.4f}")
        print(f"  - R2:   {r2:.4f}")
        
        # Registrar parámetros y métricas en MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Registrar el modelo de Scikit-Learn directamente
        mlflow.sklearn.log_model(model, "model", registered_model_name="LinearRegressionDiabetesModel")
        print("✅ ¡Modelo entrenado y registrado exitosamente en MLflow!")
        
if __name__ == '__main__':
    train_model()
