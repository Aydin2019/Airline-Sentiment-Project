# AirSentinel – Setup Instructions

## Folder Structure (required)

```
sentiment-app/
├── app.py
├── requirements.txt
├── models/
│   ├── lstm_airline_sentiment.h5
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
└── templates/
    └── index.html
```

## Steps

1. Copy your model files into the `models/` folder:
   - lstm_airline_sentiment.h5
   - tokenizer.pkl
   - label_encoder.pkl

2. Install dependencies (run once):
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://localhost:5000
   ```

## Notes
- No GloVe file needed at runtime — the model is already trained and saved.
- Ctrl+Enter in the text box also triggers analysis.
