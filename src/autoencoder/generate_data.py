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
    noise = np.random.normal(0, noise_factor, output_data.shape)

    anomaly_mask = np.random.rand(output_data.shape[0]) < anomaly_probability

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
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))

    generated_data = decoder.predict(random_latent_vectors)
    if anomaly:
        anomaly_p = 0.1
    else:
        anomaly_p = 0
    generated_data_noisy = add_output_noise(
        generated_data,
        noise_factor=0.95,
        anomaly_probability = anomaly_p, 
        high_noise_factor=3.0    
    )

    generated_data_rescaled = scaler.inverse_transform(generated_data_noisy)

    for i, output_file in enumerate(output_files):
        np.savetxt(output_file, generated_data_rescaled[:, i * 75:(i + 1) * 75], delimiter=',')

    generate_dataset(output_files[0], output_files[1], output_files[2])
    
    return add_anomaly_to_data("../../data/dataset_generated.csv", anomalies)

import pandas as pd
import random

def add_anomaly_to_data(file_path, input_files, anomaly_percentage=0.2):
    """
    Agrega bloques completos de anomalías a los datos del archivo `file_path`, seleccionando partes aleatorias
    del archivo de anomalía y agregándolos al archivo principal. Además, añade una columna `Failures` indicando
    si un dato es normal (0) o anomalía (1).

    :param file_path: Ruta al archivo CSV principal donde se agregarán las anomalías.
    :param input_files: Lista de rutas a los archivos CSV que contienen las anomalías (3 columnas: BER, OSNR, InputPower).
    :param anomaly_percentage: Porcentaje máximo del archivo principal que puede ser reemplazado por anomalías (por defecto 20%).

    :return: Número de bloques de anomalías agregados.
    """

    main_data = pd.read_csv(file_path)

    required_columns = ['BER', 'OSNR', 'InputPower']
    if not all(col in main_data.columns for col in required_columns):
        raise ValueError(f"El archivo {file_path} debe contener las columnas {required_columns}.")

    if 'Failures' not in main_data.columns:
        main_data['Failures'] = 0

    anomaly_count = 0  
    max_anomaly_rows = int(len(main_data) * anomaly_percentage)

    for input_file in input_files:
        anomaly_data = pd.read_csv(input_file)

        if not all(col in anomaly_data.columns for col in required_columns):
            raise ValueError(f"El archivo {input_file} debe contener las columnas {required_columns}.")

        anomaly_data['Failures'] = 1

        num_anomaly_rows = len(anomaly_data)
        
        if num_anomaly_rows > max_anomaly_rows:
            raise ValueError(f"El archivo de anomalía {input_file} tiene más filas que el porcentaje permitido de anomalías.")

        available_indices = list(range(len(main_data)))
        random.shuffle(available_indices) 

        for i in range(0, max_anomaly_rows, num_anomaly_rows):
            if len(available_indices) == 0:
                break

            insert_index = available_indices.pop()
            
            main_data = pd.concat([main_data.iloc[:insert_index],
                                   anomaly_data,
                                   main_data.iloc[insert_index:]], ignore_index=True)
            anomaly_count += 1

            if len(main_data) > len(main_data) * (1 + anomaly_percentage):
                break
    main_data.to_csv(file_path, index=False)

    return anomaly_count
