from pyclustering.cluster import cluster_visualizer;
from pyclustering.cluster.cure import cure;
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

data = pd.read_csv("../../data/dataset_anomaly.csv")
features = ["BER", "OSNR"]
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start_time = time.time()
cure_instance = cure(X_scaled, 3)
cure_instance.process()
clusters = cure_instance.get_clusters()
end_time = time.time() 

visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, X_scaled)
visualizer.show()

print(f"CURE detectó {len(clusters)} clústeres (excluyendo ruido).")
print(data[["BER", "OSNR"]].head())
print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

