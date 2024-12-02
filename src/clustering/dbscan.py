import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import time
import os

file_paths = ["../../data/dataset_real.csv", "../../data/dataset_generated.csv"]
features = ["BER", "OSNR", "InputPower"]
for file_path in file_paths:
    data = pd.read_csv(file_path)
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
    clean_name = os.path.splitext(os.path.basename(file_path))[0]

    # Imprimir resultados
    print(f"DBSCAN detectó {n_clusters} clústeres en {clean_name} (excluyendo ruido).")
    print(f"DBSCAN identificó {n_noise} puntos de ruido.")
    print(data[["BER", "OSNR", "DBSCAN_Cluster"]].head())
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

