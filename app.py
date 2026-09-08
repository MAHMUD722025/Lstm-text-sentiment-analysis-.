import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

st.set_page_config(page_title="LSTM Sentiment Analysis", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Bidirectional LSTM sentiment classifier")

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
        words = review.lower().split()
        seq = [[word_index.get(w, 2) + 3 for w in words]]
        padded_seq = pad_sequences(seq, maxlen=100)
        score = float(model.predict(padded_seq, verbose=0)[0][0])

        if score >= 0.5:
            st.success("Positive review 😇")
        else:
            st.error("Negative review 😡")

        st.metric("Prediction Score", f"{score:.4f}")

st.markdown("---")
st.caption("Sentiment analysis model developed by Md. Nazmul Hasan Khan Mahmud")
