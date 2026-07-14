import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, Input, SimpleRNN
from tensorflow.keras.models import Sequential, load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL = BASE_DIR / "char_rnn_many_to_many.keras"
VOCAB = BASE_DIR / "char_vocab.pkl"
DATASET = BASE_DIR / "tiny-shakespeare.txt"

SEQ_LEN = 80
STEP_SIZE = 3
EMBED_DIM = 64
RNN_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 8
TRAIN_SPLIT = 0.9


st.set_page_config(
    page_title="Shakespeare Character Generator",
    page_icon="🔴",
    layout="centered",
)


def load_text():
    with open(DATASET, "r", encoding="utf-8") as file:
        return file.read()


def build_vocabulary(text):
    characters = sorted(set(text))
    char_to_idx = {char: index for index, char in enumerate(characters)}
    return char_to_idx, characters


def encode_text(text, char_to_idx):
    return np.array([char_to_idx[char] for char in text], dtype=np.int32)


def split_input_target(chunk):
    return chunk[:, :-1], chunk[:, 1:]


def make_sequence_dataset(encoded_text, shuffle):
    if len(encoded_text) <= SEQ_LEN + 1:
        raise ValueError("Dataset is too short for the configured sequence length.")

    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=encoded_text,
        targets=None,
        sequence_length=SEQ_LEN + 1,
        sequence_stride=STEP_SIZE,
        sampling_rate=1,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )

    dataset = dataset.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(vocab_size):
    model = Sequential(
        [
            Input(shape=(SEQ_LEN,)),
            Embedding(input_dim=vocab_size, output_dim=EMBED_DIM),
            SimpleRNN(RNN_UNITS, return_sequences=True),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model():
    if not DATASET.exists():
        raise FileNotFoundError(f"{DATASET.name} was not found.")

    print("Training many-to-many character RNN...")

    text = load_text()
    char_to_idx, idx_to_char = build_vocabulary(text)
    encoded_text = encode_text(text, char_to_idx)

    split_index = int(len(encoded_text) * TRAIN_SPLIT)
    train_encoded = encoded_text[:split_index]
    val_encoded = encoded_text[split_index:]

    train_dataset = make_sequence_dataset(train_encoded, shuffle=True)
    val_dataset = make_sequence_dataset(val_encoded, shuffle=False)

    with open(VOCAB, "wb") as file:
        pickle.dump(
            {
                "char_to_idx": char_to_idx,
                "idx_to_char": idx_to_char,
                "vocab_size": len(idx_to_char),
            },
            file,
        )

    model = build_model(len(idx_to_char))

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL)

    validation_loss, validation_accuracy = model.evaluate(val_dataset, verbose=0)

    load_saved_model.clear()
    load_saved_vocab.clear()

    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")

    return {
        "history": history.history,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "vocab_size": len(idx_to_char),
        "dataset_size": len(encoded_text),
    }


@st.cache_resource
def load_saved_model():
    return load_model(MODEL)


@st.cache_resource
def load_saved_vocab():
    with open(VOCAB, "rb") as file:
        return pickle.load(file)


def sample_next_index(probabilities, temperature):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.maximum(probabilities, 1e-8)

    scaled = np.log(probabilities) / max(temperature, 0.1)
    scaled = np.exp(scaled - np.max(scaled))
    scaled = scaled / np.sum(scaled)

    return int(np.random.choice(len(scaled), p=scaled))


def generate_text(seed_text, length, temperature):
    model = load_saved_model()
    vocab = load_saved_vocab()

    char_to_idx = vocab["char_to_idx"]
    idx_to_char = vocab["idx_to_char"]
    pad_index = char_to_idx.get(" ", 0)

    if not seed_text:
        seed_text = "ROMEO:\n"

    generated_indices = [char_to_idx.get(char, pad_index) for char in seed_text]

    for _ in range(length):
        window = generated_indices[-SEQ_LEN:]
        padded = np.full((SEQ_LEN,), pad_index, dtype=np.int32)
        padded[-len(window):] = window

        predictions = model.predict(padded[np.newaxis, :], verbose=0)
        next_index = sample_next_index(predictions[0, -1], temperature)
        generated_indices.append(next_index)

    return "".join(idx_to_char[index] for index in generated_indices)


if not MODEL.exists() or not VOCAB.exists():
    train_model()


st.markdown(
    """
    <style>
    :root {
        --crimson: #E10600;
        --deep-red: #B00000;
        --dark-red: #7A0000;
        --pure-white: #FFFFFF;
        --off-white: #FFF5F5;
    }

    .stApp {
        background: linear-gradient(180deg, var(--pure-white) 0%, var(--off-white) 100%);
    }

    html, body, [class*="css"] {
        color: var(--dark-red);
    }

    h1 {
        color: var(--crimson) !important;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 0px;
    }

    h2, h3, h4 {
        color: var(--deep-red) !important;
        font-weight: 700 !important;
        border-bottom: 3px solid var(--crimson);
        padding-bottom: 6px;
    }

    .stCaption, p, label, .stMarkdown {
        color: var(--dark-red) !important;
    }

    textarea {
        background-color: var(--off-white) !important;
        border: 2px solid var(--crimson) !important;
        border-radius: 10px !important;
        color: var(--dark-red) !important;
    }

    .result-box {
        margin-top: 1rem;
        padding: 1rem 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(225, 6, 0, 0.15);
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 10px 30px rgba(176, 0, 0, 0.08);
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Shakespeare Character Generator")
st.caption("Character-level many-to-many RNN using Tiny Shakespeare")

seed_text = st.text_area(
    "Seed text",
    value="ROMEO:\n",
    height=120,
    help="Enter a few characters or a short line to continue.",
)

length = st.slider("Characters to generate", min_value=50, max_value=500, value=200, step=50)
temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1)

col1, col2 = st.columns(2)

with col1:
    generate = st.button("Generate", use_container_width=True)

with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if generate:
    with st.spinner("Generating text..."):
        output = generate_text(seed_text, length, temperature)
    st.markdown(f"<div class='result-box'>{output}</div>", unsafe_allow_html=True)