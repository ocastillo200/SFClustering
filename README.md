# Clustering para encontrar Soft Failures en redes ópticas.
La técnica utilizada se basa en reducción de dimensionalidad utilizando un autoencoder y luego una comparación de 3 algoritmos de clustering de scikitlearn (HDBSCAN, DBSCAN, CURE). Luego se evalúa su performance de manera supervisada (con fallos etiquetados). 
Los datos utilizados son de mediciones reales, pero el autoencoder también puede generar datos sintéticos para probar el clustering en grandes volumenes de datos.
