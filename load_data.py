import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_paths):
    """
    Carga y preprocesa los datos, asegurando que estén listos para el entrenamiento.
    
    Args:
        file_paths (list): Lista con las rutas de los archivos CSV.
        
    Returns:
        data_scaled (numpy.ndarray): Datos escalados con forma (num_samples, input_dim).
        scaler (StandardScaler): El objeto StandardScaler utilizado para normalizar los datos.
    """
    # Cargar los archivos CSV
    data_list = [pd.read_csv(file_path, header=None) for file_path in file_paths]
    
    # Concatenar los datos a lo largo de las columnas (por cada muestra)
    data = pd.concat(data_list, axis=1).values  # Concatenar a lo largo de las columnas (horizontal)

    # Escalar los datos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler
