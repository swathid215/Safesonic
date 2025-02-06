import os
import sounddevice as sd
import librosa
import numpy as np
import pickle
import scipy.io.wavfile as wav
import requests
from io import BytesIO
import streamlit as st
import speech_recognition as sr
import time

# Load the trained model
MODEL_PATH = "C:/Users/91876/Desktop/project/models/sound_classifier_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

MAX_PAD_LEN = 100  # Padding length for MFCC features

# Telegram Bot Details
TELEGRAM_BOT_TOKEN = "7660995028:AAHXcw6oFpWDOsgSU1oKXKPIfERlXnNiI2w"
TELEGRAM_CHAT_ID = "1809002268"

# Path to the videos for different sounds
VIDEO_PATH_BABY_CRYING = "C:/Users/91876/Desktop/project/Baby Girl Cry.mp4"
VIDEO_PATH_SIREN = "C:/Users/91876/Desktop/project/Siren.mp4"
VIDEO_PATH_GLASS_BREAKING = "C:/Users/91876/Desktop/project/GlassBreak.mp4"

# Emergency Bot Details
TELEGRAM_BOT_TOKEN_NEW = "7982558679:AAGgz_qDDGEKbXiIJzkdWm79hKbCaZVpjk8"
TELEGRAM_CHAT_ID_NEW = "5055350369"

def send_telegram_message(message, video_path=None):
    """Sends a notification to Telegram with vibration when a specific sound is detected."""
    # Send text message
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message,
        "disable_notification": False,  # Ensures message is sent as high-priority
    }
    response = requests.post(url, data=data)
    
    # Send video if it's provided (e.g., when a specific sound is detected)
    if video_path:
        with open(video_path, "rb") as video_file:
            video = video_file.read()
            files = {"video": video}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo", 
                          data={"chat_id": TELEGRAM_CHAT_ID}, 
                          files=files)
    
    if response.status_code == 200:
        st.success("Notification sent successfully!")
    else:
        st.error("Failed to send notification.")

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

# Speech Recognition Functionality
def recognize_speech():
    """Recognize speech and convert it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)  # Set timeout to avoid indefinite waiting
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.write("Could not understand audio. Please speak clearly.")
            return None
        except sr.RequestError:
            st.write("API unavailable. Please try again later.")
            return None

# Emergency Alert System
def send_emergency_alert(contact_number):
    """Send emergency alert with location to a contact number."""
    location_url = "https://maps.google.com/?q=12.9716,77.5946"  # Example coordinates (replace with actual GPS fetching)
    alert_message = f"🚨 Emergency Alert! Location: {location_url}"
    send_telegram_message(alert_message, None)

# Streamlit UI
st.title("🔊 Sound Classification Web App")
st.write("This app classifies audio from a `.wav` file located in your test samples directory.")

# Button to load and classify test audio from the specified directory
test_samples_directory = "C:/Users/91876/Desktop/project/test_samples"

# Load all .wav files from the directory
test_files = [f for f in os.listdir(test_samples_directory) if f.endswith('.wav')]

# Dropdown to select a file for classification
file_selection = st.selectbox("Select a test sample to classify", test_files)

if st.button("📤 Classify Selected Sample"):
    if file_selection:
        audio_file_path = os.path.join(test_samples_directory, file_selection)
        st.write(f"Classifying audio file: {file_selection}")

        # Extract features and make prediction
        mfcc_features, energy, zero_crossing_rate, spectral_centroid = extract_audio_features(audio_file_path)

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

            # Map detected sound to corresponding video
            video_path = None
            if sound_label == "👶 Baby Crying":
                video_path = VIDEO_PATH_BABY_CRYING
            elif sound_label == "🚨 Sirens":
                video_path = VIDEO_PATH_SIREN
            elif sound_label == "💥 Glass Breaking":
                video_path = VIDEO_PATH_GLASS_BREAKING

            # Send notification to Telegram (with video if applicable)
            send_telegram_message(
                f"🚨 Alert! {sound_label} detected from the audio file {file_selection}!",
                video_path=video_path
            )

            # Display results
            st.write(f"### Prediction: {sound_label}")
            st.write(f"**Confidence:** {confidence_score * 100:.2f}%")
            st.write(f"**Energy:** {energy}")
            st.write(f"**Zero Crossing Rate:** {zero_crossing_rate}")
            st.write(f"**Spectral Centroid:** {spectral_centroid}")
        else:
            st.error("❌ Error extracting audio features.")
    else:
        st.error("❌ Please select a valid audio file for classification.")

# Emergency Alert Section
if 'is_listening_for_alert' not in st.session_state:
    st.session_state.is_listening_for_alert = False

# Toggle for starting emergency alert mode
if st.button("🚨 Emergency Mode"):
    st.session_state.is_listening_for_alert = True

# Speech Recognition for Emergency Alert
if st.session_state.is_listening_for_alert:
    emergency_contact_number = st.text_input("Enter the emergency contact number:")
    
    if emergency_contact_number:
        if len(emergency_contact_number) == 10 and emergency_contact_number.isdigit():
            st.write("Listening for emergency command...")
            command = recognize_speech()

            if command:
                st.write(f"Command recognized: {command}")
                if "emergency" in command.lower():  # Detect if emergency is mentioned
                    send_emergency_alert(emergency_contact_number)
                    st.write("Emergency alert has been sent!")
                else:
                    st.write("No emergency detected in the command.")
        else:
            st.error("Please enter a valid 10-digit phone number.")
