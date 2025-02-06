import os
import librosa
import numpy as np
import pickle

# Paths to the test samples
TEST_SAMPLES_PATH = "../test_samples/"  # Update this path

# Load the trained model
with open("../models/sound_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

MAX_PAD_LEN = 100  # Padding length for MFCC features

# Function to extract MFCC features from an audio file
def extract_mfcc(file_path, n_mfcc=40, max_pad_len=MAX_PAD_LEN):
    try:
        # Load the audio file with a fixed sampling rate of 16kHz
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate the MFCC features to ensure consistent length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None  # Return None if processing fails

# Process and classify all test audio files
label_map = {
    1: "Baby Crying",
    2: "Sirens",
    3: "Glass Breaking"
}

for file_name in os.listdir(TEST_SAMPLES_PATH):
    if file_name.endswith(".wav"):
        file_path = os.path.join(TEST_SAMPLES_PATH, file_name)
        
        # Extract MFCC features from the test audio file
        mfcc_features = extract_mfcc(file_path)
        
        # Print the shape of MFCC features for the current file
        if mfcc_features is not None:
            print(f"Shape of extracted MFCC for {file_name}: {mfcc_features.shape}")
        
        if mfcc_features is not None:
            # Flatten the MFCC features for the model
            mfcc_features_flattened = mfcc_features.flatten().reshape(1, -1)
            
            # Predict using the trained model
            prediction = model.predict(mfcc_features_flattened)[0]
            
            # Output the prediction result
            sound_label = label_map.get(prediction, "Unknown Sound")
            print(f"{file_name}: {sound_label}\n")
