"""
Streamlit UI for the one to one character level RNN.
If a trained model is not found, it trains automatically on first launch and caches it.
Run with: streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "corpus.txt")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
MAPPINGS_PATH = os.path.join(MODEL_DIR, "mappings.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="One to One RNN", layout="centered")


@st.cache_resource(show_spinner=False)
def get_model_and_mappings():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    if os.path.exists(MODEL_PATH) and os.path.exists(MAPPINGS_PATH):
        model = load_model(MODEL_PATH)
        with open(MAPPINGS_PATH, "rb") as f:
            mappings = pickle.load(f)
        return model, mappings

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

    model = Sequential([
        SimpleRNN(64, input_shape=(1, vocab_size), activation="tanh"),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X, y, epochs=150, batch_size=32, verbose=0)

    model.save(MODEL_PATH)
    mappings = {"char_to_idx": char_to_idx, "idx_to_char": idx_to_char, "vocab_size": vocab_size}
    with open(MAPPINGS_PATH, "wb") as f:
        pickle.dump(mappings, f)

    return model, mappings


def generate_text(model, mappings, seed_char, length):
    char_to_idx = mappings["char_to_idx"]
    idx_to_char = mappings["idx_to_char"]
    vocab_size = mappings["vocab_size"]

    if seed_char not in char_to_idx:
        return None

    result = seed_char
    current = seed_char
    for _ in range(length):
        x = np.zeros((1, 1, vocab_size))
        x[0, 0, char_to_idx[current]] = 1.0
        pred = model.predict(x, verbose=0)
        next_char = idx_to_char[int(np.argmax(pred))]
        result += next_char
        current = next_char
    return result


st.title("One to One Character RNN")
st.write(
    "This is a one to one recurrent neural network. Each prediction takes a single "
    "character as input and produces a single character as output. There is no "
    "sequence unrolling inside the model. Text generation happens by feeding the "
    "predicted character back in as the next input, one step at a time."
)

with st.spinner("Loading model, training on first run only, this can take a moment"):
    model, mappings = get_model_and_mappings()

st.success("Model is ready")

st.subheader("Try it")

available_chars = sorted(mappings["char_to_idx"].keys())
st.write("Available characters in vocabulary:")
st.code("".join(available_chars))

col1, col2 = st.columns(2)

with col1:
    seed_char = st.text_input("Enter one starting character", value=available_chars[0], max_chars=1)

with col2:
    length = st.slider("How many characters to generate", min_value=10, max_value=300, value=100, step=10)

if st.button("Generate"):
    if not seed_char:
        st.error("Please enter a starting character")
    elif seed_char not in mappings["char_to_idx"]:
        st.error("That character is not in the training vocabulary. Pick one from the list above.")
    else:
        output = generate_text(model, mappings, seed_char, length)
        st.subheader("Generated text")
        st.write(output)

st.divider()



with st.expander("View training dataset"):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        st.text(f.read())
