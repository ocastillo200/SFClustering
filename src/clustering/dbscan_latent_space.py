import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
import tracemalloc

import time

def DBSCAN_latent_space(encoder, data, scaler, block_size, features, file_path):
    scaled_data = scaler.transform(data[features])

    num_features = len(features)
    expected_size = (len(scaled_data) // block_size) * block_size
    scaled_data = scaled_data[:expected_size, :]
    data = data.iloc[:expected_size] 

    scaled_data_flat = scaled_data.reshape(-1, block_size * num_features)

    latent_data = encoder(scaled_data_flat).numpy()

    latent_scaler = StandardScaler()
    latent_scaled = latent_scaler.fit_transform(latent_data)

    min_samples = 12
    epsilon = 2

    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    start_time = time.time()  
    hdbscan_model = DBSCAN(min_samples=min_samples, eps=epsilon).fit(latent_scaled)
    hdbscan_labels = hdbscan_model.labels_
    end_time = time.time()  

    snapshot_end = tracemalloc.take_snapshot()
    stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_memory_used = sum(stat.size_diff for stat in stats)

    cluster_sizes = np.bincount(hdbscan_labels[hdbscan_labels != -1]) 
    densest_cluster = np.argmax(cluster_sizes)  

    cluster_labels = {cluster: (0 if cluster == densest_cluster else 1) for cluster in np.unique(hdbscan_labels)}
    cluster_labels[-1] = 1  

    block_predictions = np.array([cluster_labels[label] for label in hdbscan_labels])

    reconstructed_labels = np.repeat(block_predictions, block_size)[:len(data)]

    if "Failures" in data.columns:
        y_true = data["Failures"].values
        y_pred = reconstructed_labels

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1score = f1_score(y_true, y_pred)

        # Resultados
        print(f"Resultados para {file_path}:")
        print(f"- Clusters detectados (excluyendo ruido): {len(cluster_sizes)}")
        print(f"- Cluster más denso: {densest_cluster}")
        print(f"- Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        print(f"- Precisión (Accuracy): {accuracy:.3f}")
        print(f"- Precisión (Precision): {precision:.3f}")
        print(f"- Sensibilidad (Recall): {recall:.3f}")
        print(f"- F1-Score: {f1score:.3f}")
        print(f"DBSCAN utilizó aproximadamente {total_memory_used / 1024:.2f} KB de memoria.")

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

file = "../../data/dataset_real.csv"
data = pd.read_csv(file)
if 'Failures' not in data.columns:
    raise ValueError("El dataset debe contener la columna 'Failures'.")
features = ["BER", "OSNR", "InputPower"]
block_size = 100
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
encoder = load_model('../autoencoder/models/encoder_model_normalized_real.keras')

DBSCAN_latent_space(encoder, data, scaler, block_size, features, file)
