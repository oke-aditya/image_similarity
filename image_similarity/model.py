__all__ = ["create_model"]

import tensorflow as tf


def create_model(img_shape: tuple = (512, 512, 3)):
    """
    Cretes a simple Convolutional Auto-encoder.
    It uses the functional API of tensorflow.
    Since pretrained models are not allowed, I'm creating one.
    Need to create some programmatic API for this.
    """

    input_ = tf.keras.layers.Input(img_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(input_)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(
        img_shape[2], (3, 3), activation="sigmoid", padding="same"
    )(x)

    # Create autoencoder model
    autoencoder = tf.keras.Model(input_, decoded)
    # input_autoencoder_shape = autoencoder.layers[0].input_shape[1:]
    # output_autoencoder_shape = autoencoder.layers[-1].output_shape[1:]

    # Create encoder model
    encoder = tf.keras.Model(input_, encoded)  # set encoder
    # input_encoder_shape = encoder.layers[0].input_shape[1:]
    # output_encoder_shape = encoder.layers[-1].output_shape[1:]

    # Create decoder model
    decoded_input = tf.keras.Input(shape=output_encoder_shape)
    decoded_output = autoencoder.layers[-7](decoded_input)  # Conv2D
    decoded_output = autoencoder.layers[-6](decoded_output)  # UpSampling2D
    decoded_output = autoencoder.layers[-5](decoded_output)  # Conv2D
    decoded_output = autoencoder.layers[-4](decoded_output)  # UpSampling2D
    decoded_output = autoencoder.layers[-3](decoded_output)  # Conv2D
    decoded_output = autoencoder.layers[-2](decoded_output)  # UpSampling2D
    decoded_output = autoencoder.layers[-1](decoded_output)  # Conv2D

    # Final decoder Model
    decoder = tf.keras.Model(decoded_input, decoded_output)
    # decoder_input_shape = decoder.layers[0].input_shape[1:]
    # decoder_output_shape = decoder.layers[-1].output_shape[1:]

    return autoencoder, encoder, decoder
