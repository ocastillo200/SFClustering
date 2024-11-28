import numpy as np

def add_output_noise(output_data, noise_factor):
    """Añadir ruido a las salidas generadas."""
    noise = np.random.normal(0, noise_factor, output_data.shape)
    noisy_output = output_data + noise
    return noisy_output


def generate_new_data(decoder, scaler, latent_dim, num_samples, output_files):
    """
    Genera nuevos datos usando el decodificador del autoencoder.
    
    Args:
        decoder (tensorflow.keras.Model): El modelo del decodificador.
        scaler (sklearn.preprocessing.StandardScaler): El objeto scaler utilizado para normalizar los datos.
        latent_dim (int): La dimensión del espacio latente.
        num_samples (int): El número de muestras a generar.
        output_files (list): Los archivos de salida para guardar los datos generados.
    """
    # Generar muestras aleatorias en el espacio latente
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    
    # Generar nuevos datos desde el espacio latente
    generated_data = decoder.predict(random_latent_vectors)

    generated_data_noisy = add_output_noise(generated_data, noise_factor=0.95)

    # Desescalar los datos generados
    generated_data_rescaled = scaler.inverse_transform(generated_data_noisy)
    
    # Guardar los datos generados en archivos CSV
    for i, output_file in enumerate(output_files):
        np.savetxt(output_file, generated_data_rescaled[:, i * 75:(i + 1) * 75], delimiter=',')
        
