import numpy as np
from sklearn.model_selection import train_test_split

def add_gaussian_noise(data, noise_factor):
    """Añade ruido gaussiano a los datos."""
    noise = np.random.normal(0, noise_factor, data.shape)
    noisy_data = data + noise
    return noisy_data

def train_autoencoder(autoencoder, data, epochs=50, batch_size=32, noise_factor=0):
    """
    Entrena el autoencoder con los datos dados, añadiendo ruido a los datos de entrada.
    
    Args:
        autoencoder (Model): Modelo de autoencoder.
        data (np.ndarray): Datos preprocesados.
        epochs (int): Número de épocas.
        batch_size (int): Tamaño del batch.
        noise_factor (float): Factor de ruido para los datos de entrada.
    """
    # Dividir datos en entrenamiento y validación
    data_train, data_val = train_test_split(data, test_size=0.2, random_state=42)
    
    # Añadir ruido gaussiano a los datos de entrada
    data_train_noisy = add_gaussian_noise(data_train, noise_factor)
    data_val_noisy = add_gaussian_noise(data_val, noise_factor)
    
    # Compilar y entrenar el autoencoder
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        data_train_noisy, data_train,  # Entrenamos el autoencoder con los datos ruidosos
        validation_data=(data_val_noisy, data_val),  # Validamos con los datos ruidosos
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True
    )
