import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import time
import os

file_paths = ["../../data/dataset_real.csv", "../../data/dataset_generated.csv"]
features = ["BER", "OSNR", "InputPower"]
for file_path in file_paths:
    data = pd.read_csv(file_path)
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    min_cluster_size = 5  
    min_samples = 5  
    start_time = time.time()
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X_scaled)
    hdbscan_labels = hdbscan_model.labels_
    end_time = time.time()

    clusters = set(hdbscan_labels)
    n_clusters = len(clusters) - (1 if -1 in hdbscan_labels else 0)
    n_noise = list(hdbscan_labels).count(-1)

    data["HDBSCAN_Cluster"] = hdbscan_labels
    clean_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"HDBSCAN detectó {n_clusters} clústeres en {clean_name} (excluyendo ruido).")
    print(f"HDBSCAN identificó {n_noise} puntos de ruido.")
    print(data[["BER", "OSNR", "HDBSCAN_Cluster"]].head())
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

