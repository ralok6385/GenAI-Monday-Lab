# CSET419 – Introduction to Generative AI
# Lab 4: Text Generation using LSTM

print("=== Lab 4 started ===")

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# DATASET
# -----------------------------
text = """
artificial intelligence is transforming modern society.
machine learning allows systems to improve automatically with experience.
deep learning uses neural networks.
"""

text = text.lower()

# -----------------------------
# CHARACTER LEVEL TOKENIZATION
# -----------------------------
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 20
X, y = [], []

for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])

X = np.array(X) / float(len(chars))
y = np.array(y)

print("Dataset prepared successfully")

# -----------------------------
# LSTM MODEL
# -----------------------------
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1)))
model.add(Dense(len(chars), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)

print("Training started...")
model.fit(
    X.reshape(X.shape[0], X.shape[1], 1),
    y,
    epochs=5,
    batch_size=16
)

print("Training completed")

# -----------------------------
# TEXT GENERATION
# -----------------------------
seed_text = text[:20]
generated_text = seed_text

for _ in range(100):
    x_pred = np.array([[char_to_idx[c] for c in seed_text]])
    x_pred = x_pred / float(len(chars))
    prediction = model.predict(x_pred.reshape(1, 20, 1), verbose=0)
    next_char = idx_to_char[np.argmax(prediction)]
    generated_text += next_char
    seed_text = seed_text[1:] + next_char

print("\nGenerated Text:\n")
print(generated_text)

print("=== Lab 4 finished ===")
