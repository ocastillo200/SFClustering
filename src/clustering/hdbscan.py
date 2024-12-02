import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import time

# Cargar el dataset desde un archivo CSV
data = pd.read_csv("../../data/dataset_normal.csv")

# Selección de características relevantes
features = ["BER", "OSNR"]
X = data[features].values

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configuración del modelo HDBSCAN
min_cluster_size = 5  # Tamaño mínimo del clúster
min_samples = 5  # Puntos mínimos para formar un clúster
start_time = time.time()
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X_scaled)
hdbscan_labels = hdbscan_model.labels_
end_time = time.time()

# Identificar clústeres y puntos de ruido
clusters = set(hdbscan_labels)
n_clusters = len(clusters) - (1 if -1 in hdbscan_labels else 0)
n_noise = list(hdbscan_labels).count(-1)

# Agregar etiquetas al dataframe original
data["HDBSCAN_Cluster"] = hdbscan_labels

# Imprimir resultados
print(f"HDBSCAN detectó {n_clusters} clústeres (excluyendo ruido).")
print(f"HDBSCAN identificó {n_noise} puntos de ruido.")
print(data[["BER", "OSNR", "HDBSCAN_Cluster"]].head())
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# Guardar los resultados en un archivo
data.to_csv("hdbscan_results.csv", index=False)

import matplotlib.pyplot as plt
import numpy as np

unique_labels = set(hdbscan_labels)
core_samples_mask = np.zeros_like(hdbscan_labels, dtype=bool)
core_samples_mask[hdbscan_model.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = hdbscan_labels == k

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.title(f"Estimated number of clusters: {n_clusters}")
plt.show()
