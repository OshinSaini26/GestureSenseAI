import os
import joblib
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# --------------------------
# Load Dataset
# --------------------------

X_train = pd.read_csv("dataset/processed/X_train.csv")
X_test = pd.read_csv("dataset/processed/X_test.csv")

y_train = pd.read_csv("dataset/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# --------------------------
# Create Neural Network
# --------------------------

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42
)

print("\nTraining model...\n")

model.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", accuracy * 100)

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

# --------------------------
# Save Model
# --------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/gesture_model.pkl")

print("\nModel Saved Successfully!")