%%writefile src/data_cleaning.py
import pandas as pd
import numpy as np
import os
import sys

def clean_and_engineer_data(input_path="data/raw_data.csv", output_path="data/clean_data.csv"):
    """Limpia el dataset crudo y genera características interactivas para el modelado."""
    print(f"Iniciando limpieza e ingeniería de características para: {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe.")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    
    # 1. Limpieza básica
    df = df.dropna()
    df = df.drop_duplicates()
    
    # 2. Ingeniería de Características (Feature Engineering)
    print("Generando variables de interacción...")
    df['bmi_age_interaction'] = df['bmi'] * df['age']
    df['bp_bmi_interaction'] = df['bp'] * df['bmi']
    df['s1_s2_ratio'] = df['s1'] / (df['s2'] + 1e-9)
    
    # Asegurar que existe la carpeta de destino
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 3. Guardar el dataset procesado listo para el entrenamiento
    df.to_csv(output_path, index=False)
    print(f"¡Datos procesados y guardados con éxito en: {output_path}!")
    return df

if __name__ == '__main__':
    clean_and_engineer_data()
