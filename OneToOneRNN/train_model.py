"""
Trains a one-to-one character level RNN.
One-to-one means: single character in, single character out, no sequence unrolling.
Run this once: python train_model.py
It saves model/model.keras and model/mappings.pkl
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "corpus.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
MAPPINGS_PATH = os.path.join(MODEL_DIR, "mappings.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_text():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_dataset(text):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    X_idx, y_idx = [], []
    for i in range(len(text) - 1):
        X_idx.append(char_to_idx[text[i]])
        y_idx.append(char_to_idx[text[i + 1]])

    X = tf.keras.utils.to_categorical(X_idx, num_classes=vocab_size)
    y = tf.keras.utils.to_categorical(y_idx, num_classes=vocab_size)
    X = X.reshape((X.shape[0], 1, vocab_size))

    return X, y, char_to_idx, idx_to_char, vocab_size


def build_model(vocab_size):
    model = Sequential([
        SimpleRNN(64, input_shape=(1, vocab_size), activation="tanh"),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    text = load_text()
    X, y, char_to_idx, idx_to_char, vocab_size = build_dataset(text)

    model = build_model(vocab_size)
    model.summary()

    model.fit(X, y, epochs=150, batch_size=32, verbose=1)

    model.save(MODEL_PATH)
    with open(MAPPINGS_PATH, "wb") as f:
        pickle.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char, "vocab_size": vocab_size}, f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved mappings to {MAPPINGS_PATH}")


if __name__ == "__main__":
    main()
