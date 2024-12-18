import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import time

def preprocess_data(data):
    if 'BER' in data.columns and 'Failures' in data.columns:
        data['BER'] = data.apply(lambda row: row['BER'] * (1e2 if row['Failures'] == 1 else 1), axis=1)
    return data

def check_data_consistency(data, block_size):
    if len(data) < block_size:
        raise ValueError(f"El dataset es demasiado pequeño para formar bloques de tamaño {block_size}")
    if len(data) % block_size != 0:
        print(f"Advertencia: El dataset no es divisible exactamente en bloques de {block_size}. Algunas filas serán descartadas.")

def load_and_split_data(file_path):
    data = pd.read_csv(file_path)
    features = ["BER", "OSNR", "InputPower", "Failures"]
    data = data[features]
    return data

def create_autoencoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) 

    autoencoder = Model(input_layer, output_layer)
    encoder = Model(input_layer, latent)
    autoencoder.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return autoencoder, encoder

def calculate_precision(reconstructed_data, original_data):
    mse_values = []
    mae_values = []

    for i in range(original_data.shape[1]):
        mse = mean_squared_error(original_data[:, i], reconstructed_data[:, i])
        mae = mean_absolute_error(original_data[:, i], reconstructed_data[:, i])
        mse_values.append(mse)
        mae_values.append(mae)

    return mse_values, mae_values

def visualize_latent_space(encoder, data, scaler, block_size, features):
    scaled_data = scaler.transform(data)

    num_features = len(features)
    expected_size = (len(scaled_data) // block_size) * block_size
    scaled_data = scaled_data[:expected_size, :]

    scaled_data_flat = scaled_data.reshape(-1, block_size * num_features)

    latent_data = encoder(scaled_data_flat).numpy()

    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_data)

    latent_df = pd.DataFrame(latent_2d, columns=["Latent Dimension 1", "Latent Dimension 2"])
    latent_df["Sample Index"] = range(len(latent_2d))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(latent_df["Latent Dimension 1"], latent_df["Latent Dimension 2"], 
                           c=latent_df["Sample Index"], cmap="viridis")
    plt.title("Visualización del espacio latente")
    plt.xlabel("Dimensión latente 1")
    plt.ylabel("Dimensión latente 2")
    plt.colorbar(scatter, label="Índice de muestra")
    plt.show()

def main(file_path, block_size, latent_dim, epochs, batch_size):

    data = load_and_split_data(file_path)
    if file_path == "../../data/dataset_real.csv":
        data = preprocess_data(data)
        model_file = 'models/autoencoder_model_normalized_real.keras'
        encoder_file = 'models/encoder_model_normalized_real.keras'
    else:
        model_file = 'models/autoencoder_model_generated.keras'
        encoder_file = 'models/encoder_model_generated.keras'
    data = data.drop(columns=['Failures'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    check_data_consistency(scaled_data, block_size)

    expected_size = (len(scaled_data) // block_size) * block_size
    scaled_data = scaled_data[:expected_size, :]
    print(f"Datos ajustados: {scaled_data.shape}")

    num_features = scaled_data.shape[1]
    blocks = scaled_data.reshape(-1, block_size, num_features)

    X_train, X_test = train_test_split(blocks, test_size=0.2, random_state=42)
    X_train = X_train.reshape((X_train.shape[0], block_size * num_features))
    X_test = X_test.reshape((X_test.shape[0], block_size * num_features))

    input_dim = X_train.shape[1]

    if os.path.exists(model_file):
        autoencoder = load_model(model_file)
    else:
        autoencoder, encoder = create_autoencoder(input_dim, latent_dim)
        start_time = time.time()
        history = autoencoder.fit(X_train, X_train, 
                                validation_data=(X_test, X_test), 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                shuffle=True)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        autoencoder.save(model_file)
        encoder.save(encoder_file)

    encoder = autoencoder.layers[1]
    scaled_data_flat = scaled_data.reshape(-1, block_size * num_features)
    reconstructed_flat = autoencoder.predict(scaled_data_flat)
    reconstructed_blocks = reconstructed_flat.reshape(-1, num_features)
    reconstructed_blocks = scaler.inverse_transform(reconstructed_blocks)

    original_data_trimmed = data.values[:reconstructed_blocks.shape[0], :]

    mse_values, mae_values = calculate_precision(reconstructed_blocks, original_data_trimmed)

    for i, feature_name in enumerate(['BER', 'OSNR', 'InputPower']):
        print(f"{feature_name} - MSE: {mse_values[i]:.4f}, MAE: {mae_values[i]:.4f}")

    plt.figure(figsize=(12, 8))
    for i, feature_name in enumerate(['BER', 'OSNR', 'InputPower']):
        plt.plot(data[feature_name][:reconstructed_blocks.shape[0]], label=f'{feature_name} Original')
        plt.plot(reconstructed_blocks[:, i], label=f'{feature_name} Reconstructed')
    plt.title('Comparación de datos originales y reconstruidos')
    plt.xlabel('Tiempos de muestreo')
    plt.ylabel('Valores de medición')
    plt.legend()
    plt.show()

    visualize_latent_space(encoder, data, scaler, block_size, ['BER', 'OSNR', 'InputPower'])
    return autoencoder, encoder, reconstructed_blocks

FILE_PATH = "../../data/dataset_real.csv" 
BLOCK_SIZE = 100  # Tamaño de los bloques de tiempo
LATENT_DIM = 20  # Dimensionalidad del espacio latente
EPOCHS = 5000
BATCH_SIZE = 32


autoencoder, encoder, reconstructed_data = main(FILE_PATH, BLOCK_SIZE, LATENT_DIM, EPOCHS, BATCH_SIZE)
