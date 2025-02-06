import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load features and labels
with open("../models/features.pkl", "rb") as f:
    features = pickle.load(f)

with open("../models/labels.pkl", "rb") as f:
    labels = pickle.load(f)

# Flatten features for machine learning (make sure features are already flattened)
features_flattened = features.reshape(features.shape[0], -1)

# Ensure features have the correct shape (i.e., 4000 features per sample)
print(f"Features shape: {features_flattened.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_flattened, labels, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
with open("../models/sound_classifier_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model training complete. Model saved as 'sound_classifier_model.pkl'.")
