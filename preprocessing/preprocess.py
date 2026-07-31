import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = "dataset/csv"
OUTPUT_PATH = "dataset/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# Read all CSV files
# -----------------------------
dataframes = []

letters = sorted(os.listdir(DATASET_PATH))

label = 0

for file in letters:

    if file.endswith(".csv"):

        filepath = os.path.join(DATASET_PATH, file)

        df = pd.read_csv(filepath, header=None)

        df["label"] = label

        dataframes.append(df)

        print(f"{file} -> Label {label} -> {len(df)} samples")

        label += 1

# -----------------------------
# Merge dataset
# -----------------------------
dataset = pd.concat(dataframes, ignore_index=True)

print("\nTotal Samples:", len(dataset))

# -----------------------------
# Split Features & Labels
# -----------------------------
X = dataset.iloc[:, :-1]

y = dataset.iloc[:, -1]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Save
# -----------------------------
X_train.to_csv("dataset/processed/X_train.csv", index=False)

X_test.to_csv("dataset/processed/X_test.csv", index=False)

y_train.to_csv("dataset/processed/y_train.csv", index=False)

y_test.to_csv("dataset/processed/y_test.csv", index=False)

print("\nDone!")
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))