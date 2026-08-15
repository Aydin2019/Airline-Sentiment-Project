# Airline Tweet Sentiment Classifier

A deep learning model that classifies airline customer tweets as positive, negative, or neutral, served through a live web app.

**What it does:** Cleans and tokenizes tweet text, embeds it using pre-trained GloVe vectors, and classifies sentiment with a Bidirectional LSTM. A Flask app loads the trained model and returns a live prediction with confidence scores.

**Result:** ~76% classification accuracy across 3 classes on ~14,600 tweets.

**Stack:** TensorFlow/Keras (BiLSTM), GloVe 100d embeddings, Flask, Python

**Run the app:** `cd frontend && pip install -r requirements.txt && python app.py`

**Note:** To retrain from scratch, download GloVe 6B 100d embeddings from nlp.stanford.edu/projects/glove and place in the project root. Not required to run the app — the trained model is included.
