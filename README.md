
# 🎭 Interactive Mind Unit (IMU)

## 🌟 Project Overview

The **Interactive Mind Unit (IMU)** is an advanced system designed to simulate emotional responses. Using a combination of traditional sentiment analysis tools (VADER), neural network models (LSTM), emotional memory, and adaptive learning, the IMU can dynamically evolve emotions over time based on user interactions and contexts. This project aims to provide a natural and human-like emotional interaction experience through real-time sentiment evaluation and emotional feedback loops.

## ✨ Key Features

- **💬 Sentiment Analysis**:
  - Integrates **VADER Sentiment Analyzer** for fast, rule-based sentiment classification.
  - Incorporates a **LSTM-based neural network model** for deep learning-based sentiment evaluation (trained on user data if available).
  
- **🧠 Emotional Memory Bank**:
  - Maintains a memory bank of past emotional states and interaction contexts to ensure consistent emotional responses.
  
- **🎭 Emotion Generation**:
  - Generates emotions dynamically based on the interaction context (e.g., casual, urgent) and user profiles (e.g., neutral, highly sensitive).
  
- **🔁 Adaptive Learning**:
  - The system adjusts its behavior by reinforcing recurring emotions and evolving emotional intensity based on past interactions.

- **⏳ Emotional Feedback Loop**:
  - Emotions evolve naturally over time, simulating the decay or amplification of emotions based on user interactions and contextual feedback.

## 🛠️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Adithianps0/IMU-Interactive-Mind-Unit-
    cd IMU
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the `emo.json` file (containing emotion definitions) is properly formatted. Example:

    ```json
    {
        "emotions": {
            "happy": {"intensity_range": [20, 100]},
            "sad": {"intensity_range": [0, 50]},
            "angry": {"intensity_range": [50, 100]}
        }
    }
    ```

4. Run the system:

    ```bash
    python imu.py
    ```

## ▶️ Running the IMU

Once started, the IMU will load the `emo.json` file and, if available, the pre-trained sentiment model (`sentiment_model.h5`). The system then evaluates user input and adjusts the emotional states based on the interaction context.

### Example usage:

```python
process_imu(interaction_context="casual", user_profile={"mood": "neutral"}, user_input="I am feeling great today!")
```

## ⚙️ Code Structure

### Main Components

- **EmotionState Class**:
  - Defines emotional states with attributes like name, intensity, and context.
  - Includes methods to evolve the intensity of the emotion over time based on feedback.
  
- **Sentiment Analysis**:
  - **VADER**: Used for quick and effective rule-based sentiment analysis.
  - **Neural Network (LSTM)**: Loaded from `sentiment_model.h5` or trained with sample data when the system is initiated.

- **Emotion Generation**:
  - The system dynamically generates emotions based on the current context and the user's emotional memory, with complexity modulated based on user profiles and context.

- **Feedback Loop**:
  - A feedback system that evolves emotional intensity, allowing strong emotions to decay and weak emotions to grow over time.

- **Adaptive Learning**:
  - Learns from emotional patterns in the memory bank and reinforces frequently occurring emotions to simulate long-term emotional adaptation.

## 🎯 Advantages & Disadvantages

| **Advantages** | **Disadvantages** |
| -------------- | ----------------- |
| - **Real-time Sentiment Analysis**: Integrates VADER and an optional LSTM-based sentiment model for evaluating user input. | - **Small Training Dataset**: The LSTM model is trained on a small dataset, limiting generalization to complex interactions. |
| - **Emotional Persistence**: The emotional memory bank allows the system to retain emotional continuity over time. | - **Limited Contextual Understanding**: The system’s understanding of context is relatively basic and may struggle with nuanced inputs. |
| - **Adaptive Learning**: Reinforces and adapts emotional responses based on recurring patterns, simulating a natural emotional evolution. | - **Fixed Emotional Complexity**: Predefined emotional complexity based on context without deep personalization beyond mood profiles. |
| - **Scalability**: The system can be extended with more emotions, larger datasets, and richer user profiles. | - **Performance**: Neural network performance may degrade if applied to large datasets without optimization. |
| - **Modular Design**: Easy to expand with more advanced sentiment analysis models, emotion types, and contextual data. | - **Resource-Intensive**: Neural network training and evaluation can be computationally expensive, requiring careful resource management. |

## 📊 Sentiment Model Training

If you wish to retrain the sentiment analysis model, update the training data in `imu.py` and run the script. The model will automatically be saved as `sentiment_model.h5` upon completion of training.

### Example training data:

```python
texts = ["I am happy", "I am sad", "I am excited", "I am angry"]
labels = [1, 0, 1, 0]
```

Once trained, the model will be used for sentiment analysis in future interactions.

## 🚀 How IMU Works

The **Interactive Mind Unit (IMU)** combines several components to produce emotionally-driven responses:

1. **Emotion Generation**:
   - The system selects an emotional response based on the current interaction context and user profile (e.g., neutral, highly sensitive).
  
2. **Sentiment Analysis**:
   - VADER is used to assess user input sentiment (positive, neutral, negative).
   - If available, the neural network model performs a deeper sentiment analysis to further refine emotional reactions.

3. **Emotional Persistence**:
   - Emotional states are retained in the memory bank and evolve naturally over time. Strong emotions may decay, while weaker emotions may intensify based on recurring stimuli.

4. **Emotional Feedback**:
   - Emotional states adjust over time, reacting to both internal memory and new user interactions.

5. **Adaptive Learning**:
   - Frequently triggered emotions are reinforced, adjusting their intensity and persistence in future interactions.

## 📌 Example Scenarios

The following are some potential use cases for the IMU:

1. **Virtual Assistants**: The IMU can be embedded in chatbots to simulate more natural and adaptive conversations with users.
  
2. **Game NPCs**: Non-playable characters (NPCs) in games can utilize the IMU to evolve their emotional states and respond more dynamically to players.

3. **Social Media Analysis**: The IMU can be adapted for sentiment analysis and emotion modeling in real-time social media interactions, helping to gauge the emotional impact of conversations.

## 🛠️ Future Improvements

- **🗂️ Expanded Dataset**: Train the LSTM model on a larger, more diverse dataset for better performance on complex or nuanced inputs.
- **💡 Enhanced Context Understanding**: Integrate more advanced Natural Language Processing (NLP) techniques to better understand and process interaction contexts.
- **🎭 Additional Emotions**: Expand the `emo.json` file with more detailed and complex emotional states.
- **🔁 Refined Learning**: Improve the adaptive learning mechanism to simulate long-term emotional changes more effectively.
- **👤 User Personalization**: Allow the system to personalize interactions more specifically based on detailed user profiles stored across sessions.

## 🙌 Giving Credit

If you use the IMU model in your projects, please consider giving credit to the original project 


## 🤝 Contributing

Contributions are welcome! Please feel free to fork the repository, submit pull requests, or open issues to suggest improvements or additional features.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
