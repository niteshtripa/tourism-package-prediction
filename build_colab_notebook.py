# -*- coding: utf-8 -*-
"""Builds a Google Colab companion notebook that clones the pushed GitHub
repo and runs the full pipeline live, so results can be seen by just
clicking 'Open in Colab' and running all cells — no local setup needed."""

import nbformat as nbf

GITHUB_USERNAME = "niteshtripa"
REPO_NAME = "tourism-package-prediction"
REPO = f"{GITHUB_USERNAME}/{REPO_NAME}"
REPO_URL = f"https://github.com/{REPO}"
STREAMLIT_APP_URL = "https://tourism-package-prediction-gqebjzdhhzrn33gpitu3mw.streamlit.app/"

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("# Tourism Package Prediction — Colab Runner\n\n"
   "This notebook clones the project's GitHub repository and runs the full "
   "MLOps pipeline (data registration -> data preparation -> model training "
   "with MLflow tracking) live, then loads the trained model and makes a "
   "sample prediction so you can see it working end to end.\n\n"
   f"- **Repository:** {REPO_URL}\n"
   f"- **Live app (already deployed):** {STREAMLIT_APP_URL}\n\n"
   "**How to run:** Runtime -> Run all. Takes about 2-3 minutes.")

md("## 1. Install dependencies")

code("!pip install -q mlflow==3.0.1 xgboost==2.1.1")

md("## 2. Clone the project repository")

code(f'!git clone https://github.com/{REPO}.git\n'
     f'%cd {REPO_NAME}')

md("## 3. Data Registration\n\n"
   "Validates that `tourism.csv` has the expected columns and prints a "
   "summary.")

code("!python tourism_project/model_building/data_register.py")

md("## 4. Data Preparation\n\n"
   "Cleans the data and creates the train/test split.")

code("!python tourism_project/model_building/prep.py")

md("## 5. Model Training with MLflow Tracking\n\n"
   "Tunes an XGBoost classifier with GridSearchCV, logs the run to MLflow, "
   "evaluates it, and saves the best model.")

code("!python tourism_project/model_building/train.py")

md("### Inspect the MLflow experiment runs")

code('import mlflow\n'
     '\n'
     'mlflow.set_tracking_uri("sqlite:///mlflow.db")\n'
     'runs = mlflow.search_runs(experiment_names=["tourism-package-prediction"])\n'
     'runs[[c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.")]]')

md("## 6. See a Live Prediction\n\n"
   "Loads the model that was just trained and saved to `tourism_project/"
   "deployment/model.joblib`, and runs it on a sample customer to show the "
   "actual output — this is the same model the Streamlit app uses.")

code('import joblib\n'
     'import pandas as pd\n'
     '\n'
     'model = joblib.load("tourism_project/deployment/model.joblib")\n'
     '\n'
     'sample_customer = pd.DataFrame([{\n'
     '    "Age": 37,\n'
     '    "TypeofContact": "Self Enquiry",\n'
     '    "CityTier": 1,\n'
     '    "DurationOfPitch": 15,\n'
     '    "Occupation": "Salaried",\n'
     '    "Gender": "Female",\n'
     '    "NumberOfPersonVisiting": 3,\n'
     '    "NumberOfFollowups": 4,\n'
     '    "ProductPitched": "Deluxe",\n'
     '    "PreferredPropertyStar": 3.0,\n'
     '    "MaritalStatus": "Single",\n'
     '    "NumberOfTrips": 3,\n'
     '    "Passport": 1,\n'
     '    "PitchSatisfactionScore": 3,\n'
     '    "OwnCar": 1,\n'
     '    "NumberOfChildrenVisiting": 1,\n'
     '    "Designation": "Manager",\n'
     '    "MonthlyIncome": 23000,\n'
     '}])\n'
     '\n'
     'prediction = model.predict(sample_customer)[0]\n'
     'probability = model.predict_proba(sample_customer)[0, 1]\n'
     '\n'
     'print("Prediction:", "Will purchase" if prediction == 1 else "Will not purchase")\n'
     'print(f"Probability of purchase: {probability:.1%}")\n'
     'sample_customer')

md("### Try your own customer\n\n"
   "Edit any value below and re-run this cell to see how the prediction "
   "changes.")

code('my_customer = pd.DataFrame([{\n'
     '    "Age": 45,\n'
     '    "TypeofContact": "Company Invited",\n'
     '    "CityTier": 2,\n'
     '    "DurationOfPitch": 20,\n'
     '    "Occupation": "Small Business",\n'
     '    "Gender": "Male",\n'
     '    "NumberOfPersonVisiting": 4,\n'
     '    "NumberOfFollowups": 5,\n'
     '    "ProductPitched": "Super Deluxe",\n'
     '    "PreferredPropertyStar": 5.0,\n'
     '    "MaritalStatus": "Married",\n'
     '    "NumberOfTrips": 2,\n'
     '    "Passport": 1,\n'
     '    "PitchSatisfactionScore": 4,\n'
     '    "OwnCar": 1,\n'
     '    "NumberOfChildrenVisiting": 2,\n'
     '    "Designation": "Senior Manager",\n'
     '    "MonthlyIncome": 35000,\n'
     '}])\n'
     '\n'
     'prediction = model.predict(my_customer)[0]\n'
     'probability = model.predict_proba(my_customer)[0, 1]\n'
     'print("Prediction:", "Will purchase" if prediction == 1 else "Will not purchase")\n'
     'print(f"Probability of purchase: {probability:.1%}")')

md("## 7. The Full Interactive App\n\n"
   "The form-based web app (same model, point-and-click interface) is "
   "already deployed and live — Streamlit apps need their own web server, "
   "so they can't run directly inside a Colab cell. Open it here:\n\n"
   f"### 🔗 {STREAMLIT_APP_URL}\n\n"
   "## 8. GitHub Actions Pipeline\n\n"
   "This entire sequence (steps 3-5 above) also runs automatically on "
   "GitHub every time code is pushed to `main` — see it here:\n\n"
   f"### 🔗 {REPO_URL}/actions")

nb["cells"] = cells

with open("Tourism_Package_Prediction_Colab.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Colab notebook written with", len(cells), "cells")
