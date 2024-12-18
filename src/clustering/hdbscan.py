import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

file_paths = ["../../data/dataset_real.csv"]
#file_paths = ["../../data/dataset_generated.csv"]
features = ["BER", "OSNR", "InputPower"]

for file_path in file_paths:
    data = pd.read_csv(file_path)

    if 'Failures' not in data.columns:
        raise ValueError(f"El archivo {file_path} debe contener la columna 'Failures'.")

    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    min_cluster_size = 7
    min_samples = 7
    epsilon = 1.2

    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()

    start_time = time.time()
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=epsilon).fit(X_scaled)
    hdbscan_labels = hdbscan_model.labels_
    end_time = time.time()

    snapshot_end = tracemalloc.take_snapshot()

    stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_memory_used = sum(stat.size_diff for stat in stats)

    data["HDBSCAN_Cluster"] = hdbscan_labels

    cluster_counts = data.groupby("HDBSCAN_Cluster")["Failures"].mean()
    normal_clusters = cluster_counts[cluster_counts < 0.5].index.tolist()
    failure_clusters = cluster_counts[cluster_counts >= 0.5].index.tolist()
    data["HDBSCAN_Label"] = data["HDBSCAN_Cluster"].apply(
        lambda x: 0 if x in normal_clusters else (1 if x in failure_clusters else -1)
    )

    valid_data = data[data["HDBSCAN_Label"] != -1]
    accuracy = accuracy_score(valid_data["Failures"], valid_data["HDBSCAN_Label"])
    precision = precision_score(valid_data["Failures"], valid_data["HDBSCAN_Label"])
    recall = recall_score(valid_data["Failures"], valid_data["HDBSCAN_Label"])
    f1score = f1_score(valid_data["Failures"], valid_data["HDBSCAN_Label"])

    name = os.path.basename(file_path)
    data.to_csv(name, index=False)

    # Resultados
    print(f"Resultados para {file_path}:")
    print(f"- Precisión (Accuracy): {accuracy:.3f}")
    print(f"- Precisión (Precision): {precision:.3f}")
    print(f"- Sensibilidad (Recall): {recall:.3f}")
    print(f"- F1-Score: {f1score:.3f}")
    print(f"HDBSCAN detectó {len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)} clústeres.")
    print(f"HDBSCAN identificó {list(hdbscan_labels).count(-1)} puntos de ruido.")
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos\n")
    print(f"HDBSCAN utilizó aproximadamente {total_memory_used / 1024:.2f} KB de memoria.")


    plt.figure(figsize=(12, 6))
    unique_clusters = sorted(data["HDBSCAN_Cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    # OSNR vs. Input Power
    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["HDBSCAN_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  
        plt.scatter(cluster_data["InputPower"], cluster_data["OSNR"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por HDBSCAN ({name})")
    plt.xlabel("Input Power")
    plt.ylabel("OSNR")
    plt.legend(loc="best", fontsize="small")

    # Etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["InputPower"], data["OSNR"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("Input Power")
    plt.ylabel("OSNR")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()

    # BER vs. OSNR
    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["HDBSCAN_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  
        plt.scatter(cluster_data["BER"], cluster_data["OSNR"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por HDBSCAN ({name})")
    plt.xlabel("BER")
    plt.ylabel("OSNR")
    plt.legend(loc="best", fontsize="small")

    # Etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["BER"], data["OSNR"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("BER")
    plt.ylabel("OSNR")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()

    # Input Power vs. BER

    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["HDBSCAN_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  
        plt.scatter(cluster_data["BER"], cluster_data["InputPower"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por HDBSCAN ({name})")
    plt.xlabel("BER")
    plt.ylabel("Input Power")
    plt.legend(loc="best", fontsize="small")

    # Etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["BER"], data["InputPower"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("BER")
    plt.ylabel("Input Power")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()
