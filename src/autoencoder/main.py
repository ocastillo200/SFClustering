from load_data import load_and_preprocess_data
from autoencoder import build_autoencoder
from train import train_autoencoder
from generate_data import generate_new_data
from tensorflow.keras.models import load_model
import os
import numpy as np

file_paths = ["../../data/normal/BER.csv", "../../data/normal/InputPower.csv", "../../data/normal/OSNR.csv"]
output_files = ["../../data/generated/generated_BER.csv", "../../data/generated/generated_InputPower.csv", "../../data/generated/generated_OSNR.csv"]

file_paths_anomaly = ["../../data/anomaly/BER_Filtro.csv", "../../data/anomaly/InputPower_Filtro.csv", "../../data/anomaly/OSNR_Filtro.csv","../../data/anomaly/BER_IASEN.csv", "../../data/anomaly/InputPower_IASEN.csv", "../../data/anomaly/OSNR_IASEN.csv" ]
output_files_anomaly = ["../../data/generated/generated_BER_Anomaly.csv", "../../data/generated/generated_InputPower_Anomaly.csv", "../../data/generated/generated_OSNR_Anomaly.csv"]

# Hiperparámetros
latent_dim = 32
num_samples = 1000
epochs = 150
batch_size = 32
model_file_1 = 'autoencoder_model.keras'
model_file_2 = 'autoencoder_model_anomaly.keras'  
anomaly = True

if anomaly:
    data_scaled, scaler = load_and_preprocess_data(file_paths_anomaly)
else:
    data_scaled, scaler = load_and_preprocess_data(file_paths)

if os.path.exists(model_file_1) and not anomaly:
    print("Cargando el modelo entrenado...")
    autoencoder = load_model(model_file_1)  
    decoder = autoencoder.get_layer('decoder')
elif os.path.exists(model_file_2) and anomaly:
    print("Cargando el modelo entrenado...")
    autoencoder = load_model(model_file_2)
    decoder = autoencoder.get_layer('decoder')
else:
    print("Entrenando el modelo...")
    input_dim = data_scaled.shape[1]  
    autoencoder, encoder, decoder = build_autoencoder(input_dim, latent_dim)
    train_autoencoder(autoencoder, data_scaled, epochs, batch_size)
    if anomaly:
        autoencoder.save(model_file_2)
        print(f"Modelo entrenado guardado en {model_file_2}")
    else:
        autoencoder.save(model_file_1)
        print(f"Modelo entrenado guardado en {model_file_1}")

std_originals = []
 
if not anomaly:
    for i, file_path in enumerate(file_paths):
        original_data = np.genfromtxt(file_path, delimiter=',')
        std_original = np.std(original_data) 
        std_originals.append(std_original)
        clean_name = os.path.basename(file_path)
        print(f"Desviación estándar de los datos originales para {clean_name}: {std_original}")
    generate_new_data(decoder, scaler, latent_dim, num_samples, output_files)
    for i, output_file in enumerate(output_files):
        generated_data = np.genfromtxt(output_file, delimiter=',')  
        std_generated = np.std(generated_data)  
        clean_name = os.path.basename(output_file)
        print(f"Desviación estándar de los datos generados para {clean_name}: {std_generated}")
        precision = (std_generated / std_originals[i]) * 100  
        print(f"Precisión de {clean_name}: {precision:.2f}%")
else:
    generate_new_data(decoder, scaler, latent_dim, num_samples, output_files_anomaly)