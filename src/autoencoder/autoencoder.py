from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim, latent_dim):
    """
    Construye un autoencoder con arquitectura simple de codificador y decodificador.
    
    Args:
        input_dim (int): Dimensión de la entrada.
        latent_dim (int): Dimensión del espacio latente.
    
    Returns:
        autoencoder (Model): Modelo completo del autoencoder.
        encoder (Model): Modelo del codificador.
        decoder (Model): Modelo del decodificador.
    """
    # Capas del codificador
    inputs = Input(shape=(input_dim,), name="encoder_input")
    encoder_layer = Dense(128, activation="relu")(inputs)
    latent_space = Dense(latent_dim, activation="relu", name="latent_space")(encoder_layer)

    # Construir el modelo codificador
    encoder = Model(inputs, latent_space, name="encoder")

    # Capas del decodificador
    latent_inputs = Input(shape=(latent_dim,), name="decoder_input")
    decoder_layer_1 = Dense(128, activation="relu")(latent_inputs)
    # Cambiar la última capa para que produzca el tamaño correcto
    decoder_output = Dense(input_dim, activation="sigmoid")(decoder_layer_1)  # Esto asegura que la salida tenga tamaño input_dim (225)
    
    # Construir el modelo decodificador
    decoder = Model(latent_inputs, decoder_output, name="decoder")

    # Conectar codificador y decodificador en el autoencoder completo
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

    return autoencoder, encoder, decoder
