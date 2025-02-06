import os
import librosa
import numpy as np
import pickle

# Paths to the datasets
DATASET_PATHS = {
    "baby_crying": "../dataset/baby_crying/",    
    "sirens": "../dataset/sirens/",
    "glass": "../dataset/glass/",
}

MAX_PAD_LEN = 100  # Padding length for MFCC
ENERGY_THRESHOLD = 0.01  # Silence detection threshold (adjust if needed)
n_mfcc = 40  # Number of MFCC coefficients

def extract_mfcc(file_path, n_mfcc=40, max_pad_len=MAX_PAD_LEN, energy_threshold=ENERGY_THRESHOLD):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        rms_energy = np.mean(librosa.feature.rms(y=audio))  # Compute RMS energy

        if rms_energy < energy_threshold:  # If silent, ignore this file
            print(f"Skipping {file_path}: Low energy detected ({rms_energy:.5f})")
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        # Flatten the MFCC features to match the expected input size for the classifier
        mfcc_flattened = mfcc.flatten()
        return mfcc_flattened
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Initialize feature and label lists
features, labels = [], []
label_map = {"baby_crying": 1, "sirens": 2, "glass": 3}

# Extract features from all datasets
for category, label in label_map.items():
    dataset_path = DATASET_PATHS[category]

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dataset_path, file_name)
            mfcc_features = extract_mfcc(file_path)

            if mfcc_features is not None:
                features.append(mfcc_features)
                labels.append(label)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Save the extracted features and labels to files
with open("../models/features.pkl", "wb") as f:
    pickle.dump(features, f)

with open("../models/labels.pkl", "wb") as f:
    pickle.dump(labels, f)

print("Preprocessing complete. Features and labels saved.")
