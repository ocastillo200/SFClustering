import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import time

#data = pd.read_csv("../utils/Devices.csv")
data = pd.read_csv("../../data/dataset_normal.csv")
features = ["BER", "OSNR", "InputPower"]
X = data[features].values

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configuración del modelo DBSCAN
start_time = time.time()    
dbscan_model = DBSCAN(eps=2, min_samples=70).fit(X_scaled)
dbscan_labels = dbscan_model.labels_
end_time = time.time()

# Identificar clústeres y puntos de ruido
clusters = set(dbscan_labels)
n_clusters = len(clusters) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

# Agregar etiquetas al dataframe original
data["DBSCAN_Cluster"] = dbscan_labels

# Imprimir resultados
print(f"DBSCAN detectó {n_clusters} clústeres (excluyendo ruido).")
print(f"DBSCAN identificó {n_noise} puntos de ruido.")
print(data[["BER", "OSNR", "DBSCAN_Cluster"]].head())
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

# Guardar los resultados en un archivo
data.to_csv("dbscan_results.csv", index=False)

import matplotlib.pyplot as plt
import numpy as np

unique_labels = set(dbscan_labels)
core_samples_mask = np.zeros_like(dbscan_labels, dtype=bool)
core_samples_mask[dbscan_model.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = dbscan_labels == k

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