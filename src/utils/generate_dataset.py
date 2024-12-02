import pandas as pd
import numpy as np
import os

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

    output_filename = f"../../data/dataset_{output_type}.csv"
    dataset.to_csv(output_filename, index=False)
    
    print(f"Dataset generado y guardado como: {os.path.basename(output_filename)}")

file1 = "../../data/anomaly/BER_Filtro.csv"  
file2 = "../../data/anomaly/OSNR_Filtro.csv" 
file3 = "../../data/anomaly/InputPower_Filtro.csv"  

generate_dataset(file1, file2, file3, output_type="anomaly")
