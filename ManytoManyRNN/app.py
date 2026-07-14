import os
import pickle
import numpy as np
import streamlit as st

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

MODEL = "nextword_model.keras"
TOKENIZER = "tokenizer.pkl"


def train_model():

    print("Training model...")

    with open("dataset.txt", "r", encoding="utf-8") as f:
        text = f.read().lower()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])

    total_words = len(tokenizer.word_index) + 1

    input_sequences = []

    lines = text.split("\n")[:5000]

    for line in lines:

        token_list = tokenizer.texts_to_sequences([line])[0]

        if len(token_list) < 2:
            continue

        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i + 1])

    max_sequence_len = max(len(seq) for seq in input_sequences)

    input_sequences = pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    )

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]

    y = to_categorical(y, num_classes=total_words)

    with open(TOKENIZER, "wb") as f:
        pickle.dump(tokenizer, f)

    model = Sequential()

    model.add(
        Embedding(
            input_dim=total_words,
            output_dim=100,
            input_length=max_sequence_len - 1
        )
    )

    model.add(
        SimpleRNN(
            150,
            return_sequences=True
        )
    )

    model.add(
        SimpleRNN(150)
    )

    model.add(
        Dense(
            total_words,
            activation="softmax"
        )
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    model.save(MODEL)

    print("Model Saved Successfully.")


@st.cache_resource
def load_resources():

    model = load_model(MODEL)

    with open(TOKENIZER, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer


def predict_next_word(seed_text):

    model, tokenizer = load_resources()

    max_sequence_len = model.input_shape[1] + 1

    token_list = tokenizer.texts_to_sequences([seed_text.lower()])[0]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(token_list, verbose=0)

    predicted_index = np.argmax(prediction)

    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return output_word


if not os.path.exists(MODEL):
    train_model()


st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="📖",
    layout="centered"
)

st.markdown("""
<style>

.block-container{
    max-width:750px;
    padding-top:2rem;
}

h1{
    text-align:center;
    font-size:42px;
    font-weight:700;
}

.subtitle{
    text-align:center;
    color:#6c757d;
    font-size:18px;
    margin-bottom:35px;
}

.result-box{
    margin-top:25px;
    background-color:#f8f9fa;
    border:1px solid #dcdcdc;
    border-radius:12px;
    padding:25px;
    text-align:center;
}

.result-title{
    font-size:18px;
    color:#4b5563;
}

.result-word{
    font-size:34px;
    font-weight:700;
    margin-top:10px;
    color:#111827;
}

</style>
""", unsafe_allow_html=True)

st.title("Many-to-Many RNN")

st.markdown(
    "<div class='subtitle'>Next Word Prediction using SimpleRNN</div>",
    unsafe_allow_html=True
)

text = st.text_input(
    "Enter a sentence",
    placeholder="Example: machine learning"
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button(
        "Predict",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "Clear",
        use_container_width=True
    )

if clear:
    st.rerun()

if predict:

    if text.strip() == "":
        st.warning("Please enter a sentence.")

    else:

        predicted_word = predict_next_word(text)

        st.markdown(
    f"""
    <div class="result-box">
        <div class="result-title">Predicted Next Word</div>
        <div class="result-word">{predicted_word.upper()}</div>
    </div>
    """,
    unsafe_allow_html=True
)