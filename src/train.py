import os
import sys

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# En el notebook, la regresión lineal con las variables de interacción obtuvo el RMSE más bajo: 52.47.
# El campeón solo se asigna si el RMSE de esta ejecución queda por debajo de 60.
RMSE_THRESHOLD = 60
REGISTERED_MODEL_NAME = "LinearRegressionDiabetesModel"


def train_model(data_path="data/clean_data.csv", model_dir="models"):
    """Entrena una regresión lineal, la registra en MLflow y promociona al campeón si cumple el umbral."""
    print(f"Iniciando el entrenamiento con el dataset: {data_path}...")

    if not os.path.exists(data_path):
        print(f"Error: El archivo {data_path} no existe. Por favor ejecuta la limpieza primero.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    os.makedirs(model_dir, exist_ok=True)

    mlflow.set_experiment("Diabetes_Progression_Experiment")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("Métricas del modelo entrenado:")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - MAE:  {mae:.4f}")
        print(f"  - R2:   {r2:.4f}")

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("rmse_threshold", RMSE_THRESHOLD)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(1),
        )

        if rmse < RMSE_THRESHOLD:
            MlflowClient().set_registered_model_alias(
                REGISTERED_MODEL_NAME,
                "champion",
                model_info.registered_model_version,
            )
            print(
                f"Campeón asignado: {REGISTERED_MODEL_NAME} "
                f"versión {model_info.registered_model_version} (RMSE {rmse:.4f} < {RMSE_THRESHOLD})."
            )
        else:
            print(
                f"Modelo registrado sin alias champion: RMSE {rmse:.4f} no es menor que {RMSE_THRESHOLD}."
            )

        return {"rmse": rmse, "mae": mae, "r2": r2, "champion": rmse < RMSE_THRESHOLD}


if __name__ == "__main__":
    train_model()
