import pandas as pd
import numpy as np
import os

# Create a dataset with generated data 

def generate_dataset(file1, file2, file3, output_type="normal"):
    """
    Genera un dataset consolidado a partir de 3 archivos CSV.
    :param file1: Ruta al archivo CSV que contiene datos para BER.
    :param file2: Ruta al archivo CSV que contiene datos para OSNR.
    :param file3: Ruta al archivo CSV que contiene datos para InputPower.
    :param output_type: Indica si el archivo generado es "normal" o "anomaly".
    """
    ber_data = pd.read_csv(file1, header=None)  
    osnr_data = pd.read_csv(file2, header=None)  
    input_power_data = pd.read_csv(file3, header=None)  

    if not (ber_data.shape == osnr_data.shape == input_power_data.shape):
        raise ValueError("Los archivos CSV no tienen las mismas dimensiones.")
    
    ber_flat = ber_data.values.flatten(order="C")
    osnr_flat = osnr_data.values.flatten(order="C")
    input_power_flat = input_power_data.values.flatten(order="C")
    
    dataset = pd.DataFrame({
        "BER": ber_flat,
        "OSNR": osnr_flat,
        "InputPower": input_power_flat
    })

    output_filename = f"../../data/dataset_generated.csv"
    dataset.to_csv(output_filename, index=False)
    
    print(f"Dataset generado y guardado como: {os.path.basename(output_filename)}")

# Create a dataset with real data from SoftFailure_dataset.csv (https://github.com/Network-And-Services/optical-failure-dataset)

def clean_and_split_csv(input_csv):
    """
    Limpia y divide un archivo CSV de datos en varios archivos separados según el tipo de información.
    
    :param input_csv: Ruta al archivo CSV de entrada que contiene los datos de dispositivos e infraestructura.
    """
    df = pd.read_csv(input_csv)

    df['Failure'] = df['Failure'].fillna(0).astype(int)

    device_data = df[df['Type'] == 'Devices'][['Timestamp', 'ID', 'BER', 'OSNR', 'Failure']]
    infrastructure_data = df[df['Type'] == 'Infrastructure'][['Timestamp', 'ID', 'InputPower', 'OutputPower', 'Failure']]

    # Get data from Ampli1

    ampli1_data = infrastructure_data[infrastructure_data['ID'] == 'Ampli1'][['Timestamp', 'InputPower']]

    if ampli1_data.empty:
        print("No se encontraron datos para Ampli1.")
        return

    device_data = pd.merge(device_data, ampli1_data, on='Timestamp', how='left')

    if 'InputPower' not in device_data.columns:
        print("La columna 'InputPower' no se agregó correctamente.")
        return

    device_data['InputPower'] = device_data['InputPower'].fillna(method='ffill')

    device_data.to_csv('../../data/dataset_real.csv', index=False)

    print("Archivo generado con éxito: 'Device_with_InputPower.csv' ")

input_csv = '../../data/labeled/SoftFailure_dataset.csv'
clean_and_split_csv(input_csv)
