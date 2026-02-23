import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from matplotlib.backends.backend_pdf import PdfPages
import os

# -------------------------
# Setup
# -------------------------
np.random.seed(237)
tf.random.set_seed(237)

# -------------------------
# Load MNIST
# -------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# -------------------------
# Parameters
# -------------------------
latent_dim = 2
batch_size = 32
epochs = 10

# -------------------------
# Encoder
# -------------------------
encoder_input = Input(shape=(28, 28, 1))

x = Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = Flatten()(x)
x = Dense(32, activation="relu")(x)

latent = Dense(latent_dim, name="latent")(x)

encoder = Model(encoder_input, latent)

# -------------------------
# Decoder
# -------------------------
decoder_input = Input(shape=(latent_dim,))
x = Dense(14 * 14 * 64, activation="relu")(decoder_input)
x = Reshape((14, 14, 64))(x)
x = Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_output = Conv2D(1, 3, activation="sigmoid", padding="same")(x)

decoder = Model(decoder_input, decoder_output)

# -------------------------
# Autoencoder (STABLE)
# -------------------------
autoencoder_output = decoder(encoder(encoder_input))
autoencoder = Model(encoder_input, autoencoder_output)

autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# -------------------------
# Train
# -------------------------
history = autoencoder.fit(
    x_train, x_train,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# -------------------------
# Save ALL outputs to PDF
# -------------------------
pdf_path = os.path.join(os.getcwd(), "vertopal.com_Lab3.pdf")

with PdfPages(pdf_path) as pdf:

    # 1. Sample images
    plt.figure(figsize=(6,6))
    idxs = [13, 690, 2375, 42013]
    for i, idx in enumerate(idxs):
        plt.subplot(2,2,i+1)
        plt.imshow(x_train[idx,:,:,0], cmap="gnuplot2")
        plt.axis("off")
    plt.suptitle("Sample MNIST Images")
    pdf.savefig()
    plt.close()

    # 2. Training loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    pdf.savefig()
    plt.close()

    # 3. Latent space
    z_test = encoder.predict(x_test)
    plt.figure(figsize=(8,8))
    sc = plt.scatter(z_test[:,0], z_test[:,1], c=y_test, cmap="brg", s=5)
    plt.colorbar(sc)
    plt.title("Latent Space Representation")
    plt.xlabel("z1")
    plt.ylabel("z2")
    pdf.savefig()
    plt.close()

    # 4. Generated manifold
    n = 20
    digit_size = 28
    figure = np.zeros((digit_size*n, digit_size*n))

    grid = np.linspace(-3, 3, n)

    for i, yi in enumerate(grid):
        for j, xi in enumerate(grid):
            z_sample = np.array([[xi, yi]])
            digit = decoder.predict(z_sample)[0].reshape(28,28)
            figure[
                i*digit_size:(i+1)*digit_size,
                j*digit_size:(j+1)*digit_size
            ] = digit

    plt.figure(figsize=(10,10))
    plt.imshow(figure, cmap="gnuplot2")
    plt.axis("off")
    plt.title("Generated Digit Manifold")
    pdf.savefig()
    plt.close()

print(f"PDF created successfully → {pdf_path}")
