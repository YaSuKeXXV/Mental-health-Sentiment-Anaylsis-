import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = tf.keras.models.load_model('mental_health_sentiment_model.keras')

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = {0: 'Risk', 1:'Worried', 2:'Neutral', 3: 'Good', 4:'Great'}

max_length = 30

def predict_sentiment(text, confidence=None):
    #preprocess
    sequence = tokenizer.texts_to_sequence([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding = 'post')

    prediction = model.predict(padded)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return {
        'input_text': text,
        'predicted_class': label_map[class_index],
        'confidence_score': round(confidence, 4)
    }