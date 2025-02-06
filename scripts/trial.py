import os
import sounddevice as sd
import librosa
import numpy as np
import pickle
import scipy.io.wavfile as wav

# Paths to the test samples
TEST_SAMPLES_PATH = "C:/Users/91876/Desktop/project/test_samples/"  # Update this path

# Load the trained model
with open("C:/Users/91876/Desktop/project/models/sound_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

MAX_PAD_LEN = 100  # Padding length for MFCC features

# Function to extract MFCC features and other audio features
def extract_audio_features(file_path, n_mfcc=40, max_pad_len=MAX_PAD_LEN):
    try:
        # Load the audio file with a fixed sampling rate of 16kHz
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Extract other audio features
        energy = np.sum(np.square(audio))  # Total energy of the audio signal
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))  # Zero crossing rate
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))  # Spectral centroid
        
        # Pad or truncate the MFCC features to ensure consistent length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc, energy, zero_crossing_rate, spectral_centroid
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None, None  # Return None if processing fails

# Function to record audio from microphone
def record_audio(filename, duration=5, sample_rate=16000):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wav.write(filename, sample_rate, audio_data)  # Save the audio as a .wav file
    print(f"Recording finished and saved as {filename}")

# Process and classify the recorded audio
label_map = {
    1: "Baby Crying",
    2: "Sirens",
    3: "Glass Breaking"
}

# Record audio (example: 5 seconds recording)
audio_filename = "recorded_audio.wav"
record_audio(audio_filename, duration=5)

# Extract audio features from the recorded audio file
mfcc_features, energy, zero_crossing_rate, spectral_centroid = extract_audio_features(audio_filename)

# Print the shape of MFCC features and other extracted audio features
if mfcc_features is not None:
    print(f"Shape of extracted MFCC for {audio_filename}: {mfcc_features.shape}")
    print(f"Energy: {energy}")
    print(f"Zero Crossing Rate: {zero_crossing_rate}")
    print(f"Spectral Centroid: {spectral_centroid}")

# Predict and classify if MFCC features were extracted
if mfcc_features is not None:
    # Flatten the MFCC features for the model
    mfcc_features_flattened = mfcc_features.flatten().reshape(1, -1)
    
    # Predict probabilities (confidence scores) using the trained model
    prediction_probabilities = model.predict_proba(mfcc_features_flattened)[0]
    
    # Get the predicted class index with the highest probability
    predicted_class = np.argmax(prediction_probabilities)
    confidence_score = prediction_probabilities[predicted_class]
    
    # Output the prediction result along with other features
    sound_label = label_map.get(predicted_class + 1, "Unknown Sound")  # Map predicted class index to label
    print(f"{audio_filename}: {sound_label}")
    print(f"Confidence: {confidence_score * 100:.2f}%")
    print(f"Energy: {energy}, Zero Crossing Rate: {zero_crossing_rate}, Spectral Centroid: {spectral_centroid}")