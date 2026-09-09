import re
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

st.set_page_config(page_title="LSTM Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Bidirectional LSTM sentiment classifier")

MAX_WORDS = 10000
MAX_LEN = 100

@st.cache_resource
def load_assets():
    model = load_model("lstm_model.keras")
    word_index = imdb.get_word_index()
    return model, word_index

model, word_index = load_assets()


def preprocess_review(text):
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())
    sequence = [1]  # Keras IMDB START token

    for word in words:
        index = word_index.get(word)
        if index is None:
            sequence.append(2)  # OOV
        else:
            keras_index = index + 3
            sequence.append(keras_index if keras_index < MAX_WORDS else 2)

    return pad_sequences([sequence], maxlen=MAX_LEN)


review = st.text_area("Enter a movie review:", height=180)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        padded_seq = preprocess_review(review)
        score = float(model.predict(padded_seq, verbose=0)[0][0])

        if score >= 0.5:
            st.success("Positive review 😇")
        else:
            st.error("Negative review 😡")

        st.metric("Prediction Score", f"{score:.4f}")

st.markdown("---")
st.caption("Sentiment analysis model developed by Md. Nazmul Hasan Khan Mahmud")
