import streamlit as st
import os
import sounddevice as sd
import librosa
import numpy as np
import pickle
import scipy.io.wavfile as wav
import requests
from io import BytesIO

# Load the trained model
MODEL_PATH = "C:/Users/91876/Desktop/project/models/sound_classifier_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MAX_PAD_LEN = 100  # Padding length for MFCC features

# Telegram Bot Details
TELEGRAM_BOT_TOKEN = "7660995028:AAHXcw6oFpWDOsgSU1oKXKPIfERlXnNiI2w"
TELEGRAM_CHAT_ID = "1809002268"

# Path to the baby crying video (MP4)
VIDEO_PATH = "C:/Users/91876/Desktop/project/Baby Girl Cry.mp4"

def send_telegram_message(message, video_path=None):
    """Sends a notification to Telegram and optionally sends a video."""
    # Send text message
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

    # Send video if it's provided
    if video_path:
        with open(video_path, "rb") as video_file:
            video = video_file.read()
            files = {"video": video}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo", 
                          data={"chat_id": TELEGRAM_CHAT_ID}, 
                          files=files)

# Function to extract MFCC and other audio features
def extract_audio_features(file_path, n_mfcc=40, max_pad_len=MAX_PAD_LEN):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        energy = np.sum(np.square(audio))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc, energy, zero_crossing_rate, spectral_centroid
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None, None, None

# Function to record audio from microphone
def record_audio(duration=5, sample_rate=16000):
    st.write("🎤 Recording audio... Please wait.")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return audio_data

# Streamlit UI
st.title("🔊 Sound Classification Web App")
st.write("This app records audio and classifies it into different categories.")

# Record audio button
if st.button("🎙 Record Audio"):
    audio_data = record_audio(duration=5)
    audio_filename = "recorded_audio.wav"
    wav.write(audio_filename, 16000, audio_data)
    st.write("✅ Recording completed!")

    # Extract features and make prediction
    mfcc_features, energy, zero_crossing_rate, spectral_centroid = extract_audio_features(audio_filename)

    if mfcc_features is not None:
        mfcc_features_flattened = mfcc_features.flatten().reshape(1, -1)
        prediction_probabilities = model.predict_proba(mfcc_features_flattened)[0]
        predicted_class = np.argmax(prediction_probabilities)
        confidence_score = prediction_probabilities[predicted_class]

        # Class labels
        label_map = {
            1: "👶 Baby Crying",
            2: "🚨 Sirens",
            3: "💥 Glass Breaking"
        }
        sound_label = label_map.get(predicted_class + 1, "❓ Unknown Sound")

        # Send notification to Telegram with both message and video if baby crying is detected
        send_telegram_message(f"🚨 Alert! {sound_label} detected!", video_path=VIDEO_PATH if sound_label == "👶 Baby Crying" else None)

        # Display results
        st.write(f"### Prediction: {sound_label}")
        st.write(f"**Confidence:** {confidence_score * 100:.2f}%")
        st.write(f"**Energy:** {energy}")
        st.write(f"**Zero Crossing Rate:** {zero_crossing_rate}")
        st.write(f"**Spectral Centroid:** {spectral_centroid}")

    else:
        st.error("❌ Error extracting audio features.")
