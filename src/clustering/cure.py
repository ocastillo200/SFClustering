from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
import os
import numpy as np
import matplotlib.pyplot as plt

#file_paths = ["../../data/dataset_real.csv"]
file_paths = ["../../data/dataset_generated.csv"]
features = ["BER", "OSNR", "InputPower"]

for file_path in file_paths:
    data = pd.read_csv(file_path)
    
    if 'Failures' not in data.columns:
        raise ValueError(f"El archivo {file_path} debe contener la columna 'Failures'.")

    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start_time = time.time()
    cure_instance = cure(X_scaled, 30, 10)  
    cure_instance.process()
    clusters = cure_instance.get_clusters() 
    end_time = time.time() 

    labels = [-1] * len(X)  
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id

    data["CURE_Cluster"] = labels

    normal_cluster = data["CURE_Cluster"].value_counts().idxmax()
    data["CURE_Label"] = data["CURE_Cluster"].apply(lambda x: 0 if x == normal_cluster else 1)

    accuracy = accuracy_score(data["Failures"], data["CURE_Label"])
    precision = precision_score(data["Failures"], data["CURE_Label"])
    recall = recall_score(data["Failures"], data["CURE_Label"])
    f1 = f1_score(data["Failures"], data["CURE_Label"])

    name = os.path.basename(file_path)
    data.to_csv(name, index=False)

    print(f"Resultados para {file_path}:")
    print(f"- Precisión (Accuracy): {accuracy:.3f}")
    print(f"- Precisión (Precision): {precision:.3f}")
    print(f"- Sensibilidad (Recall): {recall:.3f}")
    print(f"- F1-Score: {f1:.3f}")
    print(f"CURE detectó {len(clusters)} clústeres en {name}.")
    print(f"CURE detectó {labels.count(-1)} puntos de ruido.")
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos\n")

    plt.figure(figsize=(12, 6))

    # Crear un mapa de colores para los clusters
    unique_clusters = sorted(data["CURE_Cluster"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    cluster_color_map = {cluster: color for cluster, color in zip(unique_clusters, colors)}


    # Gráfico de clústeres
    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["CURE_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  # Diferenciar ruido con 'x'
        plt.scatter(cluster_data["BER"], cluster_data["InputPower"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por CURE ({name})")
    plt.xlabel("BER")
    plt.ylabel("Input Power")
    plt.legend(loc="best", fontsize="small")

    # Gráfico de etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["BER"], data["InputPower"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("BER")
    plt.ylabel("Input Power")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()

    # Gráfico de clústeres
    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["CURE_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  # Diferenciar ruido con 'x'
        plt.scatter(cluster_data["BER"], cluster_data["OSNR"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por CURE ({name})")
    plt.xlabel("BER")
    plt.ylabel("OSNR")
    plt.legend(loc="best", fontsize="small")

    # Gráfico de etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["BER"], data["OSNR"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("BER")
    plt.ylabel("OSNR")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()

    # Gráfico de clústeres
    plt.subplot(1, 2, 1)
    for cluster, color in cluster_color_map.items():
        cluster_data = data[data["CURE_Cluster"] == cluster]
        marker = "x" if cluster == -1 else "o"  # Diferenciar ruido con 'x'
        plt.scatter(cluster_data["InputPower"], cluster_data["OSNR"], c=[color], label=f"Cluster {cluster}", marker=marker, s=10)
    plt.title(f"Clasificación por CURE ({name})")
    plt.xlabel("Input Power")
    plt.ylabel("OSNR")
    plt.legend(loc="best", fontsize="small")

    # Gráfico de etiquetas reales
    plt.subplot(1, 2, 2)
    plt.scatter(data["InputPower"], data["OSNR"], c=data["Failures"], cmap="viridis", s=10)
    plt.title(f"Etiquetas reales (Failures) ({name})")
    plt.xlabel("Input Power")
    plt.ylabel("OSNR")
    plt.colorbar(label="Failures")

    plt.tight_layout()
    plt.show()
