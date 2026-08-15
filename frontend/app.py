import os
import re
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_airline_sentiment.h5")
TOK_PATH   = os.path.join(BASE_DIR, "models", "tokenizer.pkl")
LE_PATH    = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
MAX_LEN    = 55

# ── Load artifacts once at startup ──────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)

with open(TOK_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LE_PATH, "rb") as f:
    le = pickle.load(f)

print("Ready.")

# ── Text cleaning (same pipeline as training) ────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"won't",  "will not", text)
    text = re.sub(r"can't",  "cannot",   text)
    text = re.sub(r"n't",    " not",     text)
    text = re.sub(r"'re",    " are",     text)
    text = re.sub(r"'s",     " is",      text)
    text = re.sub(r"'d",     " would",   text)
    text = re.sub(r"'ll",    " will",    text)
    text = re.sub(r"'ve",    " have",    text)
    text = re.sub(r"'m",     " am",      text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+",   "",         text)
    text = re.sub(r"#(\w+)", r"\1",      text)
    text = re.sub(r"[!?]{2,}", " multiexclaim ", text)
    text = re.sub(r"!",      " exclaim ", text)
    text = re.sub(r"\?",     " question ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ",  text)
    text = re.sub(r"\s+",    " ",        text).strip()
    return text

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data.get("tweet", "").strip()

    if not tweet:
        return jsonify({"error": "No tweet provided"}), 400

    cleaned  = clean_text(tweet)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs    = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label    = le.inverse_transform([pred_idx])[0]

    scores = {cls: round(float(p) * 100, 1) for cls, p in zip(le.classes_, probs)}

    return jsonify({
        "sentiment": label,
        "confidence": round(float(probs[pred_idx]) * 100, 1),
        "scores": scores
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
