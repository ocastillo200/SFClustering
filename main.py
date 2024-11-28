from load_data import load_and_preprocess_data
from autoencoder import build_autoencoder
from train import train_autoencoder
from generate_data import generate_new_data
from tensorflow.keras.models import load_model
import os
import numpy as np

file_paths = ["data/normal/BER.csv", "data/normal/InputPower.csv", "data/normal/OSNR.csv"]
output_files = ["generated_BER.csv", "generated_InputPower.csv", "generated_OSNR.csv"]

# Hiperparámetros
latent_dim = 32
num_samples = 1000
epochs = 150
batch_size = 32
model_file = 'autoencoder_model.keras'  

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
for i, file_path in enumerate(file_paths):
    original_data = np.genfromtxt(file_path, delimiter=',')
    std_original = np.std(original_data) 
    std_originals.append(std_original)
    print(f"Desviación estándar de los datos originales para {file_path}: {std_original}")
generate_new_data(decoder, scaler, latent_dim, num_samples, output_files)
for i, output_file in enumerate(output_files):
    generated_data = np.genfromtxt(output_file, delimiter=',')  
    std_generated = np.std(generated_data)  
    print(f"Desviación estándar de los datos generados para {output_file}: {std_generated}")
    precision = (std_generated / std_originals[i]) * 100  
    print(f"Precisión de {output_file}: {precision:.2f}%")
