import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.generate_dataset import generate_dataset


def add_output_noise(output_data, noise_factor, anomaly_probability=0.1, high_noise_factor=3.0):
    """
    Añadir ruido a las salidas generadas, con una probabilidad de generar datos más ruidosos (anomalías).
    
    Args:
        output_data (np.ndarray): Datos de salida generados.
        noise_factor (float): Factor de ruido normal.
        anomaly_probability (float): Probabilidad de generar ruido más intenso (anomalías).
        high_noise_factor (float): Factor de ruido elevado para anomalías.
    
    Returns:
        np.ndarray: Datos con ruido añadido.
    """
    # Generar ruido normal
    noise = np.random.normal(0, noise_factor, output_data.shape)
    
    # Identificar las muestras con anomalías
    anomaly_mask = np.random.rand(output_data.shape[0]) < anomaly_probability
    
    # Añadir ruido más alto a las muestras seleccionadas
    high_noise = np.random.normal(0, high_noise_factor, output_data.shape)
    noise[anomaly_mask, :] = high_noise[anomaly_mask, :]
    
    noisy_output = output_data + noise
    return noisy_output

def generate_new_data(decoder, scaler, latent_dim, num_samples, output_files, anomaly = False, anomalies = []):
    """
    Genera nuevos datos usando el decodificador del autoencoder.
    
    Args:
        decoder (tensorflow.keras.Model): El modelo del decodificador.
        scaler (sklearn.preprocessing.StandardScaler): El objeto scaler utilizado para normalizar los datos.
        latent_dim (int): La dimensión del espacio latente.
        num_samples (int): El número de muestras a generar.
        output_files (list): Los archivos de salida para guardar los datos generados.
    """
    # Generar muestras aleatorias en el espacio latente
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    
    # Generar nuevos datos desde el espacio latente
    generated_data = decoder.predict(random_latent_vectors)
    if anomaly:
        anomaly_p = 0.1
    else:
        anomaly_p = 0
    # Añadir ruido con posibilidad de anomalías
    generated_data_noisy = add_output_noise(
        generated_data,
        noise_factor=0.95,
        anomaly_probability = anomaly_p, 
        high_noise_factor=3.0    
    )

    # Desescalar los datos generados
    generated_data_rescaled = scaler.inverse_transform(generated_data_noisy)
    
    # Guardar los datos generados en archivos CSV
    for i, output_file in enumerate(output_files):
        np.savetxt(output_file, generated_data_rescaled[:, i * 75:(i + 1) * 75], delimiter=',')
    
    generate_dataset(output_files[0], output_files[1], output_files[2])
    return add_anomaly_to_data("../../data/dataset_generated.csv", anomalies)

import pandas as pd
import random

def add_anomaly_to_data(file_path, input_files, anomaly_percentage=0.2):
    """
    Agrega bloques completos de anomalías a los datos del archivo `file_path` seleccionando partes aleatorias
    del archivo de anomalía y agregándolos al archivo principal.
    
    :param file_path: Ruta al archivo CSV principal donde se agregarán las anomalías.
    :param input_files: Lista de rutas a los archivos CSV que contienen las anomalías (3 columnas: BER, OSNR, InputPower).
    :param anomaly_percentage: Porcentaje máximo del archivo principal que puede ser reemplazado por anomalías (por defecto 20%).
    
    :return: Número de bloques de anomalías agregados.
    """
    # Leer el archivo principal
    main_data = pd.read_csv(file_path)
    
    # Asegurarse de que el archivo principal tenga las columnas correctas
    if not all(col in main_data.columns for col in ['BER', 'OSNR', 'InputPower']):
        raise ValueError(f"El archivo {file_path} debe contener las columnas 'BER', 'OSNR', 'InputPower'.")

    # Variable para contar los bloques de anomalías agregados
    anomaly_count = 0
    
    # Calcular el número máximo de filas que se pueden agregar (20% del total)
    max_anomaly_rows = int(len(main_data) * anomaly_percentage)
    
    # Iterar sobre los archivos de anomalías
    for input_file in input_files:
        anomaly_data = pd.read_csv(input_file)
        
        # Asegurarse de que el archivo de anomalía tenga las columnas correctas
        if not all(col in anomaly_data.columns for col in ['BER', 'OSNR', 'InputPower']):
            raise ValueError(f"El archivo {input_file} debe contener las columnas 'BER', 'OSNR', 'InputPower'.")
        
        # Calcular cuántas filas tiene el archivo de anomalía
        num_anomaly_rows = len(anomaly_data)
        
        # Verificar que el número de filas de anomalía no exceda el 20% del archivo principal
        if num_anomaly_rows > max_anomaly_rows:
            raise ValueError(f"El archivo de anomalía {input_file} tiene más filas que el porcentaje permitido de anomalías.")
        
        # Seleccionar índices aleatorios para insertar los bloques de anomalías
        available_indices = list(range(len(main_data)))
        random.shuffle(available_indices)  # Mezclar los índices para seleccionar aleatoriamente
        
        # Insertar el bloque de anomalías
        for i in range(0, max_anomaly_rows, num_anomaly_rows):
            if len(available_indices) == 0:
                break

            # Elegir una posición aleatoria en el archivo principal para insertar las anomalías
            insert_index = available_indices.pop()  # Elegir un índice aleatorio
            
            # Insertar el bloque completo de anomalías
            main_data = pd.concat([main_data.iloc[:insert_index],
                                   anomaly_data,
                                   main_data.iloc[insert_index:]], ignore_index=True)
            anomaly_count += 1

            # Si el número de anomalías alcanzó el límite del 20%, salir
            if len(main_data) > len(main_data) * (1 + anomaly_percentage):
                break

    # Guardar el archivo con las anomalías añadidas
    main_data.to_csv(file_path, index=False)
    
    # Retornar el número de bloques de anomalías añadidos
    return anomaly_count
