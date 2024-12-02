from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
import os

file_paths = ["../../data/dataset_real.csv", "../../data/dataset_generated.csv"]


for file_path in file_paths:
    data = pd.read_csv(file_path)
    features = ["BER", "OSNR", "InputPower"]
    X = data[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    start_time = time.time()
    cure_instance = cure(X_scaled, 3)
    cure_instance.process()
    clusters = cure_instance.get_clusters()
    end_time = time.time() 

    clean_name = os.path.splitext(os.path.basename(file_path))[0]

    print(f"CURE detectó {len(clusters)} clústeres en {clean_name} (excluyendo ruido).")
    print(data[["BER", "OSNR", "InputPower"]].head())
    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

