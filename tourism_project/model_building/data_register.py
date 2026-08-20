"""
Registers the tourism dataset: verifies it has the expected schema and
prints a short summary. Run from the repo root:

    python tourism_project/model_building/data_register.py
"""

import os
import sys

import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]


def register_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Make sure tourism.csv "
            "has been added to the tourism_project/data folder."
        )

    df = pd.read_csv(data_path)

    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    print("Dataset registered successfully")
    print(f"Path            : {data_path}")
    print(f"Rows            : {df.shape[0]}")
    print(f"Columns         : {df.shape[1]}")
    print(f"Target column   : ProdTaken")
    print(f"Class balance   :\n{df['ProdTaken'].value_counts(normalize=True).round(3)}")
    print(f"Missing values  : {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows  : {int(df.duplicated().sum())}")

    return df


if __name__ == "__main__":
    try:
        register_dataset()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Data registration failed: {exc}", file=sys.stderr)
        sys.exit(1)
