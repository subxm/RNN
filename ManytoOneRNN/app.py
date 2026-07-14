# RNN - many to one
# Dataset: spam.csv

import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "spam_model.keras")
TOKENIZER = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_WORDS = 5000
MAX_LEN = 50


# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# TRAIN MODEL
def train_model():
    print("Training model...")

    df = pd.read_csv(os.path.join(BASE_DIR, "spam.csv"), encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels into numbers
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # CLEAN SMS
    df["message"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["message"])

    sequences = tokenizer.texts_to_sequences(df["message"])

    x = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"]

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    # Save Tokenizer
    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = Sequential()

    # Embedding Layer
    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=32,
            input_length=MAX_LEN
        )
    )

    # Simple RNN Layer
    model.add(SimpleRNN(32, activation="relu"))

    # Output Layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    # Save Model
    model.save(MODEL)

    # Evaluate
    loss, accuracy = model.evaluate(x_test, y_test)

    print("\nAccuracy:", accuracy)

    # Prediction
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)

    # Confusion Matrix
    print(confusion_matrix(y_test, y_pred))


# Prediction Function
def predict_sms(message):
    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability > 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability


# Train only if model/tokenizer doesn't exist
if not os.path.exists(MODEL) or not os.path.exists(TOKENIZER):
    train_model()


# Streamlit UI
st.title("Spam Detection using RNN")
st.write("Many to One RNN Example")

message = st.text_area("Enter your message")

if st.button("Predict"):
    prediction, probability = predict_sms(message)
    st.success(prediction)
    st.write(
        "Confidence:",
        round(probability * 100, 2),
        "%"
    )