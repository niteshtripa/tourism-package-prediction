"""
Cleans the tourism dataset and creates train/test splits. Run from the
repo root:

    python tourism_project/model_building/prep.py

Writes Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to the repo root so the
GitHub Actions workflow can upload them as a job artifact for the next job.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET_COLUMN = "ProdTaken"

# Identifier columns carry no predictive signal and must be dropped before
# modeling.
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df = df.drop_duplicates()

    # "Fe Male" is a data-entry typo for "Female".
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # "Unmarried" and "Single" describe the same marital status category.
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    return df


def prepare_data(data_path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(data_path)
    df = clean_data(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Cleaned dataset shape : {df.shape}")
    print(f"Train shape           : {Xtrain.shape}")
    print(f"Test shape            : {Xtest.shape}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")

    return Xtrain, Xtest, ytrain, ytest


if __name__ == "__main__":
    prepare_data()
