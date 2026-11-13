%%writefile run_pipeline.py
import sys

# Importar las funciones de nuestros módulos en src/
from src.data_extraction import extract_data
from src.data_validation import validate_data
from src.data_cleaning import clean_and_engineer_data
from src.train import train_model

def run_complete_pipeline():
    print("==================================================")
    print("🚀 INICIANDO EJECUCIÓN DEL PIPELINE DE MLOPS")
    print("==================================================\n")
    
    # Paso 1: Extracción
    try:
        extract_data(output_path="data/raw_data.csv")
    except Exception as e:
        print(f"❌ Error en la extracción de datos: {e}")
        sys.exit(1)
        
    # Paso 2: Validación
    print("\n--- Ejecutando Paso 2: Validación de datos ---")
    validation_success = validate_data(file_path="data/raw_data.csv")
    if not validation_success:
        print("❌ El pipeline se detuvo debido a un fallo en la validación.")
        sys.exit(1)
        
    # Paso 3: Limpieza e Ingeniería de Características
    print("\n--- Ejecutando Paso 3: Limpieza y Feature Engineering ---")
    try:
        clean_and_engineer_data(input_path="data/raw_data.csv", output_path="data/clean_data.csv")
    except Exception as e:
        print(f"❌ Error en el procesamiento de datos: {e}")
        sys.exit(1)
        
    # Paso 4: Entrenamiento y Registro
    print("\n--- Ejecutando Paso 4: Entrenamiento de Modelo y MLflow ---")
    try:
        train_model(data_path="data/clean_data.csv", model_dir="models")
    except Exception as e:
        print(f"❌ Error en el entrenamiento del modelo: {e}")
        sys.exit(1)
        
    print("\n==================================================")
    print("🎉 ¡PIPELINE EJECUTADO CON ÉXITO DE EXTREMO A EXTREMO!")
    print("==================================================")

if __name__ == "__main__":
    run_complete_pipeline()
