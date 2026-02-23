import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# USER INPUTS
# ======================
dataset_choice = input("Enter dataset (mnist/fashion): ").lower()
epochs = int(input("Enter epochs: "))
batch_size = int(input("Enter batch size: "))
noise_dim = int(input("Enter noise dimension: "))
learning_rate = float(input("Enter learning rate: "))
save_interval = int(input("Save images every how many epochs?: "))

# ======================
# LOAD DATASET
# ======================
if dataset_choice == "mnist":
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
elif dataset_choice == "fashion":
    (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
else:
    raise ValueError("Invalid dataset")

x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)

# ======================
# CREATE FOLDERS
# ======================
os.makedirs(f"final_generated_images_{dataset_choice}", exist_ok=True)
os.makedirs(f"generated_samples_{dataset_choice}", exist_ok=True)

# ======================
# GENERATOR
# ======================
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_shape=(noise_dim,)),
        layers.LeakyReLU(),
        layers.Dense(512),
        layers.LeakyReLU(),
        layers.Dense(28 * 28, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return model

# ======================
# DISCRIMINATOR
# ======================
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(512),
        layers.LeakyReLU(),
        layers.Dense(256),
        layers.LeakyReLU(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy()
gen_opt = tf.keras.optimizers.Adam(learning_rate)
disc_opt = tf.keras.optimizers.Adam(learning_rate)

# ======================
# TRAIN STEP
# ======================
@tf.function
def train_step(images):
    noise = tf.random.normal([tf.shape(images)[0], noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)
        gen_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        disc_loss = (
            loss_fn(tf.ones_like(real_output), real_output) +
            loss_fn(tf.zeros_like(fake_output), fake_output)
        )

    gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grad, generator.trainable_variables))
    disc_opt.apply_gradients(zip(disc_grad, discriminator.trainable_variables))
    return gen_loss, disc_loss

# ======================
# SAVE IMAGES
# ======================
def save_images(epoch):
    noise = tf.random.normal([25, noise_dim])
    images = (generator(noise, training=False) + 1) / 2
    plt.figure(figsize=(5,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(images[i,:,:,0], cmap="gray")
        plt.axis("off")
    plt.savefig(f"generated_samples_{dataset_choice}/epoch_{epoch}.png")
    plt.close()

# ======================
# TRAIN LOOP
# ======================
for epoch in range(1, epochs + 1):
    for batch in dataset:
        g_loss, d_loss = train_step(batch)
    print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss:.2f} | G_loss: {g_loss:.2f}")
    if epoch % save_interval == 0:
        save_images(epoch)

# ======================
# FINAL IMAGES
# ======================
noise = tf.random.normal([100, noise_dim])
final_images = (generator(noise, training=False) + 1) / 2

for i in range(100):
    plt.imshow(final_images[i,:,:,0], cmap="gray")
    plt.axis("off")
    plt.savefig(f"final_generated_images_{dataset_choice}/img_{i}.png")
    plt.close()

print("GAN Training Completed Successfully")
