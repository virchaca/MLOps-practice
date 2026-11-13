
%%writefile src/data_validation.py
import pandas as pd
import numpy as np
import os
import sys

def validate_data(file_path="data/raw_data.csv"):
    """Carga el archivo CSV y realiza pruebas de validación de calidad de datos."""
    print(f"Iniciando validación de datos para: {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe. ¡Ejecuta la extracción primero!")
        sys.exit(1)
        
    df = pd.read_csv(file_path)
    
    try:
        # 1. Verificar si target existe
        assert 'target' in df.columns, "¡La columna objetivo 'target' no existe!"

        # 2. Verificar columnas requeridas
        required_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        for col in required_features:
            assert col in df.columns, f"¡La columna requerida {col} está ausente!"

        # 3. Asegurar que todos los valores de las columnas sean numéricos
        assert df.select_dtypes(include=[np.number]).shape[1] == df.shape[1], "¡Se detectaron datos no numéricos!"

        # 4. Asegurar que no existan valores nulos críticos
        assert df.isnull().sum().sum() == 0, "¡Se encontraron valores nulos críticos en el dataset!"

        # 5. Asegurar consistencia en el tamaño mínimo del dataset
        assert len(df) > 0, "¡El dataset está completamente vacío!"
        
        print("✅ ¡Validación de datos aprobada con éxito!")
        return True
        
    except AssertionError as e:
        print(f"❌ Validación fallida: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    validate_data()
