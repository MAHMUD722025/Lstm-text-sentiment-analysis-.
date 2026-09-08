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

review = st.text_area("Enter a movie review:", height=180)

if st.button("Analyze"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        # Match Keras IMDB's original indexing scheme:
        # 0 = padding, 1 = start, 2 = OOV, actual words start at 3.
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", review.lower())
        sequence = []

        for word in words:
            index = word_index.get(word)
            if index is None or index + 3 >= MAX_WORDS:
                sequence.append(2)  # OOV
            else:
                sequence.append(index + 3)

        padded_seq = pad_sequences([sequence], maxlen=MAX_LEN)
        score = float(model.predict(padded_seq, verbose=0)[0][0])

        if score >= 0.5:
            st.success("Positive review 😇")
        else:
            st.error("Negative review 😡")

        st.metric("Prediction Score", f"{score:.4f}")

st.markdown("---")
st.caption("Sentiment analysis model developed by Md. Nazmul Hasan Khan Mahmud")
