Sound Classification & Emergency Alert System

Overview: This project is a sound classification system designed to detect specific sounds such as baby crying, ambulance sirens, and more. Upon detecting these sounds, the system triggers an emergency alert that can be sent to a user-defined contact along with location information. Additionally, the project includes a frontend component using Three.js for visualization.

Features

->Sound Classification: Detects predefined sounds using machine learning.

->Emergency Alert: Sends alerts with location details to emergency contacts via Telegram.

->Frontend Visualization: Uses Three.js for an interactive UI.

->Preprocessing: Extracts features like MFCC from audio files.

->Model Training: Uses a Random Forest classifier for sound detection.

->Integration: Connects backend classification with a user-friendly interface.

->Real-Time Detection: Running app.py enables real-time sound classification and alerting.

Tech Stack

Frontend: HTML, CSS, JavaScript, Three.js

Backend: Python, Streamlit

Machine Learning: NumPy, Pandas, Librosa, Scikit-learn

Notifications: Telegram API

Database: MongoDB (if needed for storing user data)

Project Structure

📂 SAFESONIC_extracted/ 📄 app.py 📄 app1.py 📄 Baby Girl Cry.mp4 📄 contact.py 📄 GlassBreak.mp4 📄 message.py 📄 recorded_audio.wav 📄 Siren.mp4 📂 dataset/ 📂 baby_crying/ 📄 (multiple .wav files) 📂 glass/ 📄 (multiple .wav files) 📂 sirens/ 📄 (multiple .wav files) 📂 frontend/ 📄 index.html 📂 static/ 📄 notification.js 📄 threejs_scene.js 📂 models/ 📄 features.pkl 📄 labels.pkl 📄 sound_classifier_model.pkl 📂 scripts/ 📄 output_audio.wav 📄 preprocess.py 📄 recorded_audio.wav 📄 test_model.py 📄 train_model.py 📄 trial.py 📂 test_samples/ 📄 (sample .wav files)

Setup Instructions

Clone the Repository:

git clone https://github.com/your-repo.git cd SAFESONIC

Install Dependencies:

pip install numpy pandas librosa scikit-learn streamlit python-telegram-bot

Run Preprocessing & Training:

python scripts/preprocess.py python scripts/train_model.py

Run the Application (Frontend UI):

streamlit run app1.py

Run Real-Time Sound Detection:

python app.py

Usage

->Upload an audio file for classification.

->If a detected sound matches the trained categories, an alert is triggered and sent via Telegram.

->Running app.py enables real-time sound detection and alerting.

Future Enhancements

Improve model accuracy with more datasets.

Add real-time sound detection.

Implement SMS/email notifications for emergency alerts.

Expand Three.js visualization for better UI effects.
