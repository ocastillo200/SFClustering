from load_data import load_and_preprocess_data
from autoencoder import build_autoencoder
from train import train_autoencoder
from generate_data import generate_new_data
from tensorflow.keras.models import load_model
import os
import numpy as np

file_paths = ["../../data/normal/BER.csv", "../../data/normal/OSNR.csv", "../../data/normal/InputPower.csv"]
output_files = ["../../data/generated/generated_BER.csv", "../../data/generated/generated_OSNR.csv", "../../data/generated/generated_InputPower.csv"]

file_paths_anomaly_IASEN = ["../../data/anomaly/BER_IASEN.csv", "../../data/anomaly/OSNR_IASEN.csv", "../../data/anomaly/InputPower_IASEN.csv" ]
output_files_anomaly_IASEN = ["../../data/generated/generated_BER_Anomaly_IASEN.csv", "../../data/generated/generated_OSNR_Anomaly_IASEN.csv", "../../data/generated/generated_InputPower_Anomaly_IASEN.csv"]

file_paths_anomaly_Filter = ["../../data/anomaly/BER_Filtro.csv", "../../data/anomaly/OSNR_Filtro.csv", "../../data/anomaly/InputPower_Filtro.csv"]
output_files_anomaly_Filter = ["../../data/generated/generated_BER_Anomaly_Filter.csv", "../../data/generated/generated_OSNR_Anomaly_Filter.csv", "../../data/generated/generated_InputPower_Anomaly_Filter.csv"]

file_path_anomalies = ["../../data/anomaly/dataset_anomaly_Filter.csv", "../../data/anomaly/dataset_anomaly_IASEN.csv"]

# Hiperparámetros
latent_dim = 64
num_samples = 2000
epochs = 1000
batch_size = 32
model_file = 'models/autoencoder_model.keras'
anomaly = False
anomaly_type = 2

if anomaly:
    num_samples = 100
    if anomaly_type == 1:
        model_file = 'autoencoder_model_anomaly_IASEN.keras'  
        data_scaled, scaler = load_and_preprocess_data(file_paths_anomaly_IASEN)
        output_files = output_files_anomaly_IASEN
    else:
        model_file = 'autoencoder_model_anomaly_Filter.keras'
        data_scaled, scaler = load_and_preprocess_data(file_paths_anomaly_Filter)
        output_files = output_files_anomaly_Filter
else:
    data_scaled, scaler = load_and_preprocess_data(file_paths)

if os.path.exists(model_file):
    print("Cargando el modelo entrenado...")
    autoencoder = load_model(model_file)  
    decoder = autoencoder.get_layer('decoder')
else:
    print("Entrenando el modelo...")
    input_dim = data_scaled.shape[1]  
    autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
    train_autoencoder(autoencoder, data_scaled, epochs, batch_size)
    autoencoder.save(model_file)
    print(f"Modelo entrenado guardado en {model_file}")
   
std_originals = []
 
if not anomaly:
    for i, file_path in enumerate(file_paths):
        original_data = np.genfromtxt(file_path, delimiter=',')
        std_original = np.std(original_data) 
        std_originals.append(std_original)
        clean_name = os.path.basename(file_path)
        print(f"Desviación estándar de los datos originales para {clean_name}: {std_original}")
    anomalies = generate_new_data(decoder, scaler, latent_dim, num_samples, output_files, anomalies=file_path_anomalies)
    print(f"Datos generados con {anomalies} anomalías introducidas.")
    for i, output_file in enumerate(output_files):
        generated_data = np.genfromtxt(output_file, delimiter=',')  
        std_generated = np.std(generated_data)  
        clean_name = os.path.basename(output_file)
        print(f"Desviación estándar de los datos generados para {clean_name}: {std_generated}")
        precision = (std_generated / std_originals[i]) * 100  
        print(f"Precisión de {clean_name}: {precision:.2f}%")
else:
    generate_new_data(decoder, scaler, latent_dim, num_samples, output_files, anomaly)