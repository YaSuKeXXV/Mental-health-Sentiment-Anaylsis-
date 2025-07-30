import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model("mental_health_sentiment_model.keras")

# Load the saved tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define label map
label_map = {
    0: "Risk",
    1: "Worried",
    2: "Neutral",
    3: "Good",
    4: "Great"
}

# Set max length (same as used during training)
MAX_LENGTH = 30


# Prediction function
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction = model.predict(padded)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        "input": text,
        "predicted_class": label_map[class_index],
        "confidence": round(confidence, 4)
    }


# 👇 Test Examples
test_inputs = [
    "I can't do this anymore. I want it all to stop.",
    "I feel kinda nervous and stressed lately.",
    "Not bad, just a typical day.",
    "I finished my to do list and feel good about it.",
    "Honestly, I love my life right now!"
]

# Run predictions
for text in test_inputs:
    result = predict_sentiment(text)
    print(result)
