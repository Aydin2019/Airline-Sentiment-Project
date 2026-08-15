# Airline Tweet Sentiment Classifier

A deep learning NLP system that classifies airline customer tweets as **positive**, **negative**, or **neutral**, served through a live Flask web app. Built on a stacked Bidirectional LSTM with pre-trained GloVe embeddings.

## What it does

Given a tweet about an airline, the model cleans and tokenizes the text, embeds it with GloVe vectors, and predicts sentiment across three classes with confidence scores. A Flask app loads the trained model and returns live predictions through a browser interface.

## Results

Trained and evaluated on the ~14,600-tweet Twitter US Airline Sentiment dataset (2,196-tweet test set):

| Metric | Score |
|--------|-------|
| Accuracy | 76.0% |
| Weighted F1 | 0.77 |
| Macro F1 | 0.72 |

Per-class F1: negative **0.84**, positive **0.72**, neutral **0.60**. The neutral class is the hardest to separate — an expected pattern in sentiment work, since neutral language overlaps with both poles.

## Model architecture

A stacked BiLSTM with global max pooling and dropout regularization (~2.4M parameters):

- **Embedding** — GloVe 100d, 20k vocabulary (frozen, non-trainable)
- **Bidirectional LSTM** (128 units) → **Bidirectional LSTM** (64 units)
- **Global Max Pooling**
- **Dense (128) → Dropout → Dense (64) → Dropout → Dense (3, softmax)**

## Tech stack

TensorFlow/Keras, GloVe 100d embeddings, Flask, scikit-learn, NumPy, Python

## Project structure

├── frontend/
│ ├── app.py # Flask app — loads model, serves predictions
│ ├── models/ # trained model, tokenizer, label encoder
│ └── templates/ # web UI
├── notebooks/ # training notebook
├── data/ # dataset
├── outputs/ # metrics, classification report, figures
└── README.md


## Run the app

```bash
cd frontend
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 and enter a tweet.

## Retraining (optional)

The trained model is included, so you don't need GloVe to run the app. To retrain from scratch, download GloVe 6B 100d embeddings from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove) and run the notebook in `notebooks/`.

---
*Group project — 3-class NLP sentiment classification.*
