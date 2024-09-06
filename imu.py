import json
import random
import time
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


try:
    with open('emo.json', 'r') as file:
        emotion_data = json.load(file)["emotions"]
    logging.info("Emotion data loaded successfully.")
except FileNotFoundError:
    logging.error("emo.json file not found. Please ensure the file exists.")
    emotion_data = {}
except json.JSONDecodeError:
    logging.error("Failed to decode emo.json. Please check the JSON format.")
    emotion_data = {}

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Neural network for sentiment analysis
# Create a simple model if none exists
def create_model():
    model = Sequential([
        Embedding(input_dim=5000, output_dim=32, input_length=100),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training data (mock example)
def train_model(model, texts, labels):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    labels = np.array(labels)
    model.fit(padded_sequences, labels, epochs=5, batch_size=32)
    model.save('sentiment_model.h5')
    return tokenizer

# Check if a model file exists and load it, else create and train a new model
try:
    sentiment_model = load_model('sentiment_model.h5')
    logging.info("Neural network sentiment model loaded successfully.")
except:
    logging.info("No existing model found, creating and training a new model.")
    sentiment_model = create_model()
    texts = ["I am happy", "I am sad", "I am excited", "I am angry"]
    labels = [1, 0, 1, 0]
    tokenizer = train_model(sentiment_model, texts, labels)

# Emotional Memory Bank to store past emotional states with contexts
emotional_memory_bank = []

# Emotional state persistence
class EmotionState:
    def __init__(self, name, intensity=0, context=None):
        self.name = name
        self.intensity = intensity
        self.context = context

    def evolve(self, change):
        self.intensity = max(0, min(100, self.intensity + change))

    def __repr__(self):
        return f"{self.name}: {self.intensity} (Context: {self.context})"

# Dynamic Emotional Complexity
def get_emotion_complexity(interaction_context, user_profile=None):
    context_levels = {
        "casual": "simple",
        "routine": "simple",
        "urgent": "complex",
        "personal": "complex",
        "intense": "complex"
    }
    if user_profile and user_profile.get('mood') == 'highly_sensitive':
        return "complex"
    return context_levels.get(interaction_context, "medium")

# Generate emotion based on context, memory, and user profile
def generate_emotion(interaction_context, current_emotions, user_profile=None):
    try:
        complexity = get_emotion_complexity(interaction_context, user_profile)
        if complexity == "simple":
            return random.choice(list(emotion_data.keys()))
        relevant_emotions = [emotion.name for emotion in current_emotions if emotion.intensity > 50]
        return random.choice(relevant_emotions) if relevant_emotions else random.choice(list(emotion_data.keys()))
    except IndexError:
        logging.error("Emotion data is empty or not loaded correctly.")
        return "neutral"

# Emotional Feedback Loop - evolve emotions naturally over time
def feedback_loop(current_emotions):
    for emotion in current_emotions:
        if emotion.intensity > 70:
            emotion.evolve(-10)
        elif emotion.intensity < 30:
            emotion.evolve(5)

# Update Emotional Memory Bank with context
def update_memory_bank(new_emotion):
    try:
        if len(emotional_memory_bank) >= 10:
            emotional_memory_bank.pop(0)
        emotional_memory_bank.append(new_emotion)
    except Exception as e:
        logging.error(f"Error updating memory bank: {e}")

# Adaptive Learning with Reinforcement
def adaptive_learning():
    global emotional_memory_bank
    if not emotional_memory_bank:
        return
    emotion_counts = {}
    for emotion in emotional_memory_bank:
        if emotion.name not in emotion_counts:
            emotion_counts[emotion.name] = 0
        emotion_counts[emotion.name] += 1
    for emotion_name, count in emotion_counts.items():
        if count > 3:
            for emotion in emotional_memory_bank:
                if emotion.name == emotion_name:
                    emotion.evolve(5)

# Sentiment Analysis for Social Contextual Understanding
def analyze_sentiment(text):
    try:
        sentiment_score = analyzer.polarity_scores(text)['compound']
        if sentiment_score > 0.1:
            return "positive"
        elif sentiment_score < -0.1:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return "neutral"

# Neural network-based sentiment analysis
def analyze_sentiment_nn(text):
    if not sentiment_model:
        return analyze_sentiment(text)
    try:
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=100)
        prediction = sentiment_model.predict(padded_sequences)[0]
        return "positive" if prediction > 0.5 else "negative" if prediction < 0.5 else "neutral"
    except Exception as e:
        logging.error(f"Error in neural network sentiment analysis: {e}")
        return analyze_sentiment(text)

# Model Temporal Dynamics and Emotional Sequences
def model_temporal_dynamics(current_emotions, new_emotion):
    for emotion in current_emotions:
        if emotion.name == new_emotion.name:
            emotion.evolve(5)

# Main IMU process
def process_imu(interaction_context, user_profile=None, user_input=None):
    try:
        current_emotions = [
            EmotionState(name='neutral', intensity=50),
            EmotionState(name='happy', intensity=30),
            EmotionState(name='sad', intensity=20)
        ]
        if user_input:
            sentiment = analyze_sentiment_nn(user_input)
            logging.info(f"User Input Sentiment: {sentiment}")
            if sentiment == "positive":
                for emotion in current_emotions:
                    if emotion.name == 'happy':
                        emotion.evolve(10)
            elif sentiment == "negative":
                for emotion in current_emotions:
                    if emotion.name == 'sad':
                        emotion.evolve(10)
        new_emotion_name = generate_emotion(interaction_context, current_emotions, user_profile)
        new_emotion = EmotionState(name=new_emotion_name, intensity=random.randint(20, 80), context=interaction_context)
        feedback_loop(current_emotions)
        update_memory_bank(new_emotion)
        adaptive_learning()
        model_temporal_dynamics(current_emotions, new_emotion)
        logging.info(f"Context: {interaction_context}")
        logging.info(f"New Emotion: {new_emotion}")
        logging.info(f"Current Emotional States: {current_emotions}")
        logging.info(f"Emotional Memory Bank: {emotional_memory_bank}\\n")
        return new_emotion
    except Exception as e:
        logging.error(f"An error occurred during the IMU process: {e}")
        return None

# Example usage of the IMU
contexts = ["casual", "urgent", "personal", "routine", "intense"]
user_profiles = [
    {"mood": "normal"},
    {"mood": "highly_sensitive"},
    {"mood": "neutral"}
]

for _ in range(5):
    interaction_context = random.choice(contexts)
    user_profile = random.choice(user_profiles)
    user_input = "I feel great today!" if random.random() > 0.5 else "I am feeling down."
    process_imu(interaction_context, user_profile, user_input)
    time.sleep(1)