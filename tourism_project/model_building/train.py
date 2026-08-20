"""
Tunes an XGBoost classifier on the tourism train/test split, tracks the
experiment with MLflow, evaluates the best model, and saves it into
tourism_project/deployment/ so the pipeline can commit it to the repo.

Run from the repo root:

    python tourism_project/model_building/train.py
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.05, 0.1, 0.2],
}

MODEL_DIR = os.path.join("tourism_project", "deployment")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = make_column_transformer(
        (SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        (categorical_transformer, CATEGORICAL_FEATURES),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    return make_pipeline(preprocessor, model)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("tourism-package-prediction")

    Xtrain, Xtest, ytrain, ytest = load_splits()

    # Counteract the class imbalance (~80/20) so the minority "purchased"
    # class isn't ignored by the model.
    scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    pipeline = build_pipeline(scale_pos_weight)

    with mlflow.start_run():
        search = GridSearchCV(
            pipeline,
            PARAM_GRID,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        search.fit(Xtrain, ytrain)

        best_model = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        ypred = best_model.predict(Xtest)
        yproba = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "f1_score": f1_score(ytest, ypred),
            "roc_auc": roc_auc_score(ytest, yproba),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model", serialization_format="pickle")

        print("Best params:", search.best_params_)
        print("Test metrics:", metrics)
        print(classification_report(ytest, ypred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
