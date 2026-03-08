'''
This file prepares the data to be used for ML models.

There are 3 different csv files from this dataset, this script will be using
diabetes_binary_health_indicators_BRFSS2015.csv which is a clean dataset, but
without any artificial imbalance.

Install libraries:
pip install -r requirements.txt
'''

import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = "../data_processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "../data/diabetes_binary_health_indicators_BRFSS2015.csv"

TARGET = "Diabetes_binary"

RANDOM_SEED = 42

# This saves data to the folder listed in OUTPUT_DIR. It saves it as a PyTorch file
# since that makes it easier to load later with other scripts.
def save_tensor_dataset(X, y, filename):

    data = {
        "X": X,
        "y": y
    }

    path = os.path.join(OUTPUT_DIR, filename)

    torch.save(data, path)

    print(f"Saved {filename}")

# Load dataset from csv. Alternatively we could use import kagglehub
# and follow the instructions on the dataset page
def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Shape:", df.shape)
    print(df.head())

    return df

# This is just to show imbalance and missing values if we want to handle them
def inspect_data(df):

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df[TARGET].value_counts(normalize=True))

# Split into features and labels
def split_features(df):

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y

# Normalize the features that don't have binary labels
def normalize_numeric_features(X):

    numeric_cols = [
        "BMI",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "Age",
        "Education",
        "Income"
    ]

    scaler = StandardScaler()

    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, scaler

# Create train, test, validation sets (subject to change)
def create_splits(X, y):

    # 70 30 split for train and test, set aside the temp sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # use the temp sets to split again for 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def to_tensor(X, y):

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor


def main():

    df = load_data()

    inspect_data(df)

    X, y = split_features(df)

    X, scaler = normalize_numeric_features(X)

    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)

    X_train, y_train = to_tensor(X_train, y_train)
    X_val, y_val = to_tensor(X_val, y_val)
    X_test, y_test = to_tensor(X_test, y_test)

    print("\nTensor shapes:")
    print(X_train.shape, y_train.shape)
    
    save_tensor_dataset(X_train, y_train, "train.pt")
    save_tensor_dataset(X_val, y_val, "val.pt")
    save_tensor_dataset(X_test, y_test, "test.pt")


if __name__ == "__main__":
    main()