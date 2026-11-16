# Predicción de progresión de diabetes

Sistema MLOps básico para un problema de regresión. El flujo parte de un notebook exploratorio, prepara los datos, entrena un modelo, lo registra en MLflow y sirve el modelo campeón con una API.

## Objetivo

Predecir la progresión de la diabetes un año después de la medición basal a partir de variables numéricas del paciente. El repositorio separa la preparación de datos, el entrenamiento, el registro del modelo y el servicio de predicción, y automatiza el pipeline con GitHub Actions.

## Dataset

Se usa el dataset **Diabetes** de scikit-learn (`sklearn.datasets.load_diabetes`): 442 pacientes y 10 variables ya normalizadas.

| Columna | Significado |
|---|---|
| `age` | Edad |
| `sex` | Sexo |
| `bmi` | Índice de masa corporal |
| `bp` | Presión arterial media |
| `s1`–`s6` | Medidas de suero |
| `target` | Progresión de la enfermedad (variable a predecir) |

Durante la limpieza se añaden tres variables: `bmi * age`, `bp * bmi` y `s1 / s2`.

El análisis exploratorio está en `notebooks/Proyecto_MLOps.ipynb`. Ahí se compararon regresión lineal, Random Forest y Gradient Boosting, con y sin las variables de interacción. El RMSE más bajo fue el de la **regresión lineal con interacciones: 52.47**. Ese es el modelo que entrena el pipeline.

## Estructura

```text
notebooks/Proyecto_MLOps.ipynb   Análisis exploratorio
src/data_extraction.py           Carga el dataset y guarda data/raw_data.csv
src/data_validation.py           Comprueba columnas, tipos, nulos y target
src/data_cleaning.py             Limpia y crea las variables de interacción
src/train.py                     Entrena, registra en MLflow y asigna champion
src/app.py                       API FastAPI del modelo campeón
run_pipeline.py                  Ejecuta el pipeline completo en local
.github/workflows/ci.yml         Pipeline automático en cada push
requirements.txt                 Dependencias
```

`data/`, `mlruns/` y `mlflow.db` se generan al ejecutar el pipeline.

## Instalación

Hace falta Python 3.12 o superior. Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

## Pipeline de datos

Un solo comando ejecuta extracción, validación, limpieza y entrenamiento:

```bash
python run_pipeline.py
```

También se puede lanzar cada etapa:

```bash
python -m src.data_extraction
python -m src.data_validation
python -m src.data_cleaning
python -m src.train
```

La validación comprueba que exista `target`, que estén las 10 variables, que todas las columnas sean numéricas, que no haya nulos y que el dataset no esté vacío. Si falla, el pipeline se detiene.

La limpieza elimina nulos y duplicados, convierte cada columna a número y crea las tres interacciones. El resultado queda en `data/clean_data.csv`.

## Entrenamiento y registro

`src/train.py` entrena una `LinearRegression` con una partición 80/20 y `random_state=42`. En MLflow, dentro del experimento `Diabetes_Progression_Experiment`, registra:

- parámetros: tipo de modelo, `test_size`, `random_state` y el umbral de RMSE
- métricas: RMSE, MAE y R²
- el modelo, con el nombre `LinearRegressionDiabetesModel`

Para ver los experimentos en local:

```bash
mlflow ui
```

La interfaz queda en `http://127.0.0.1:5000`.

## Modelo campeón

El alias `champion` solo se asigna si el **RMSE es menor que 60**.

El umbral queda por encima del RMSE 52.47 del notebook. En la última ejecución local el RMSE fue **52.4652**, así que se asignó el alias a `LinearRegressionDiabetesModel` versión 3. Si una ejecución no baja de 60, el run queda registrado y el alias no cambia.

La API carga siempre `models:/LinearRegressionDiabetesModel@champion`.

## API

Hay que haber ejecutado el pipeline al menos una vez, para que exista un campeón. Desde la raíz del repositorio:

```bash
python -m uvicorn src.app:app --reload
```

La API queda en `http://127.0.0.1:8000`. La documentación interactiva está en `http://127.0.0.1:8000/docs`.

### GET /health

Comprueba que el servicio está en marcha. Responde `200` y `{"status": "ok"}` aunque el modelo no esté cargado.

```bash
curl http://127.0.0.1:8000/health
```

### GET /model-info

Devuelve el nombre, la versión, el alias `champion` y metadatos del run (identificador, estado y origen).

```bash
curl http://127.0.0.1:8000/model-info
```

### POST /predict

Recibe las 10 variables originales en JSON. La API vuelve a calcular las tres interacciones, predice y devuelve la predicción junto con el nombre y la versión del modelo.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.038075906433423026, "sex": 0.05068011873981862, "bmi": 0.061696206518683294, "bp": 0.0218723855140367, "s1": -0.04422349842444599, "s2": -0.03482076283769895, "s3": -0.04340084565202491, "s4": -0.002592261998183278, "s5": 0.019907486170462722, "s6": -0.01764612515980379}'
```

Ejemplo de respuesta:

```json
{
  "prediction": 151.2,
  "model_name": "LinearRegressionDiabetesModel",
  "model_version": "2"
}
```

El número de `prediction` depende del modelo cargado.

## Errores de la API

| Situación | Código |
|---|---|
| Petición válida | `200` |
| Falta un campo o el tipo no es numérico | `422` |
| No hay modelo campeón cargado | `503` |
| El modelo está cargado, pero la predicción falla | `500` |

Ejemplo de entrada inválida (`422`):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": "texto"}'
```

Si se arranca la API sin haber entrenado y registrado un campeón, `/model-info` y `/predict` responden `503`. `/health` sigue respondiendo `200`.

## CI/CD

`.github/workflows/ci.yml` se ejecuta en cada push a `main` o `master`. En un runner de Ubuntu instala `requirements.txt` y lanza, en este orden:

1. Extracción
2. Validación
3. Limpieza
4. Entrenamiento, registro y selección del campeón

Si una etapa falla, las siguientes no se ejecutan.
