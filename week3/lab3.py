# Week 3 – Variational Autoencoder (VAE)
# Course: CSET-419 – Introduction to Generative AI

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Dataset (Choose one)
# =========================
USE_FASHION_MNIST = True  # Set False for MNIST

if USE_FASHION_MNIST:
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
else:
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# =========================
# Parameters
# =========================
LATENT_DIM = 2
BATCH_SIZE = 128
EPOCHS = 20

# =========================
# Sampling (Reparameterization Trick)
# =========================
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# =========================
# Encoder
# =========================
encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# =========================
# Decoder
# =========================
latent_inputs = layers.Input(shape=(LATENT_DIM,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# =========================
# VAE Model
# =========================
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# =========================
# Training
# =========================
vae.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, None)
)

# =========================
# Reconstruction Output
# =========================
def show_reconstruction(model, data):
    z_mean, _, _ = model.encoder(data[:10])
    reconstructed = model.decoder(z_mean)

    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(data[i].squeeze(), cmap="gray")
        plt.axis("off")

        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle("Original (Top) vs Reconstructed (Bottom)")
    plt.show()

show_reconstruction(vae, x_test)

# =========================
# Generate New Images
# =========================
def generate_images(decoder, n=20):
    z_sample = tf.random.normal(shape=(n, LATENT_DIM))
    generated = decoder(z_sample)

    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, 10, i + 1)
        plt.imshow(generated[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")

    plt.suptitle("Generated Images from Latent Space")
    plt.show()

generate_images(decoder)
