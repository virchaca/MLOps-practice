import os
os.makedirs('src', exist_ok=True)

%%writefile src/data_extraction.py
import pandas as pd
from sklearn.datasets import load_diabetes
import os

def extract_data(output_path="data/raw_data.csv"):
    """Carga el dataset de diabetes de sklearn y lo guarda como un archivo CSV."""
    print("Iniciando extracción de datos...")
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target
    
    # Asegurar que existe la carpeta destino
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar en CSV
    df.to_csv(output_path, index=False)
    print(f"¡Datos extraídos con éxito y guardados en: {output_path}!")
    return df

if __name__ == "__main__":
    extract_data()
