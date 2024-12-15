import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model

import time

def process_latent_space(encoder, data, scaler, block_size, features, file_path):
    # Escalar datos originales (solo las columnas de características)
    scaled_data = scaler.transform(data[features])

    # Ajustar tamaño para ser divisible por el bloque
    num_features = len(features)
    expected_size = (len(scaled_data) // block_size) * block_size
    scaled_data = scaled_data[:expected_size, :]
    data = data.iloc[:expected_size]  # Asegurar que `data` también sea divisible

    # Aplanar los datos en bloques
    scaled_data_flat = scaled_data.reshape(-1, block_size * num_features)

    # Generar representaciones en el espacio latente
    latent_data = encoder(scaled_data_flat).numpy()

    # Normalizar el espacio latente
    latent_scaler = StandardScaler()
    latent_scaled = latent_scaler.fit_transform(latent_data)

    # Aplicar HDBSCAN en el espacio latente
    min_cluster_size = 10  # Ajusta según tus necesidades
    min_samples = 5
    epsilon = 0.15

    start_time = time.time()  # Medir tiempo de inicio
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=epsilon).fit(latent_scaled)
    hdbscan_labels = hdbscan_model.labels_
    end_time = time.time()  # Medir tiempo de fin

    # Determinar densidad de los clústeres
    cluster_sizes = np.bincount(hdbscan_labels[hdbscan_labels != -1])  # Excluir ruido
    densest_cluster = np.argmax(cluster_sizes)  # Cluster más denso

    # Etiquetar clusters: 0 para el más denso, 1 para otros y ruido
    cluster_labels = {cluster: (0 if cluster == densest_cluster else 1) for cluster in np.unique(hdbscan_labels)}
    cluster_labels[-1] = 1  # Ruido siempre etiquetado como 1

    # Asignar etiquetas a los bloques basadas en los clusters
    block_predictions = np.array([cluster_labels[label] for label in hdbscan_labels])

    # Reconstruir etiquetas por bloque para los datos originales
    reconstructed_labels = np.repeat(block_predictions, block_size)[:len(data)]

    # Calcular métricas comparando con las etiquetas originales
    if "Failures" in data.columns:
        y_true = data["Failures"].values
        y_pred = reconstructed_labels

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)

        # Imprimir resultados
        print(f"Resultados para {file_path}:")
        print(f"- Clusters detectados (excluyendo ruido): {len(cluster_sizes)}")
        print(f"- Cluster más denso: {densest_cluster}")
        print(f"- Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        print(f"- Precisión (Accuracy): {accuracy:.3f}")
        print(f"- Precisión (Precision): {precision:.3f}")
        print(f"- Sensibilidad (Recall): {recall:.3f}")
        print(f"- F1-Score: {f1score:.3f}")

    # Visualización del espacio latente
    plt.figure(figsize=(12, 6))
    unique_clusters = np.unique(hdbscan_labels)
    for cluster in unique_clusters:
        cluster_data = latent_data[hdbscan_labels == cluster]
        marker = "x" if cluster == -1 else "o"
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title("Clusters en el espacio latente")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(loc="best", fontsize="small")
    plt.show()



# Aplicar al dataset generado
file = "../../data/dataset_generated.csv"
data = pd.read_csv(file)
if 'Failures' not in data.columns:
    raise ValueError("El dataset debe contener la columna 'Failures'.")
features = ["BER", "OSNR", "InputPower"]
block_size = 100
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
encoder = load_model('../autoencoder/encoder_model_generated.keras')

process_latent_space(encoder, data, scaler, block_size, features, file)
