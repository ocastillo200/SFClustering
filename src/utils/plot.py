import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_entire_file(file_path, total_time=600):
    """
    Lee un archivo CSV y devuelve los datos concatenados y el eje de tiempo.
    """
    data = pd.read_csv(file_path, header=None)
    
    # Concatenar las columnas
    concatenated_data = data.to_numpy().flatten()
    
    # Crear eje de tiempo
    num_samples = concatenated_data.shape[0]
    time_per_sample = total_time / num_samples
    time_axis = np.linspace(0, total_time, num_samples)
    
    return time_axis, concatenated_data

def plot_sample(file_path, column_index, total_time=600):
    """
    Lee una columna específica de un archivo CSV y grafica su variación en el tiempo.
    
    Args:
        file_path (str): Ruta al archivo CSV.
        column_index (int): Índice de la columna a graficar (comienza en 0).
        total_time (int): Tiempo total de la medición (en segundos).
    """
    # Leer el archivo CSV
    data = pd.read_csv(file_path, header=None)
    
    # Seleccionar la columna deseada
    if column_index >= data.shape[1]:
        raise ValueError(f"El archivo no tiene una columna en el índice {column_index}.")
    column_data = data.iloc[:, column_index].to_numpy()
    
    # Crear eje de tiempo
    num_samples = column_data.shape[0]
    time_axis = np.linspace(0, total_time, num_samples)

    return time_axis, column_data
    

# Archivos CSV
file_path1 = "../../data/anomaly/BER_Filtro.csv"
file_path2 = "../../data/normal/BER.csv"
#file_path2 = "../../data/generated/generated_InputPower.csv"

# Leer datos
time1, data1 = plot_entire_file(file_path1)
time2, data2 = plot_entire_file(file_path2)

#time1, data1 = plot_sample(file_path1, 1)
#time2, data2 = plot_sample(file_path2, 1)

# Primer gráfico
plt.figure(1)
plt.plot(time1, data1, label="Archivo 1", color="blue")
plt.title("Medición Datos Reales")
plt.xlabel("Tiempo (s)")
plt.ylabel("Medición")
plt.legend()
plt.grid(True)

# Segundo gráfico
plt.figure(2)
plt.plot(time2, data2, label="Archivo 2", color="green")
plt.title("Medición Datos Generados")
plt.xlabel("Tiempo (s)")
plt.ylabel("Medición")
plt.legend()
plt.grid(True)

# Mostrar gráficos
plt.show()
