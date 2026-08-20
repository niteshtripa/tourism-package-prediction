# -*- coding: utf-8 -*-
"""Builds the submission notebook (Tourism_Package_Prediction.ipynb) from
the original template, replacing the Colab/PyGithub-in-notebook flow with
one that matches how this project was actually built and pushed."""

import json

import nbformat as nbf

GITHUB_USERNAME = "niteshtripa"
REPO_NAME = "tourism-package-prediction"
REPO = f"{GITHUB_USERNAME}/{REPO_NAME}"
REPO_URL = f"https://github.com/{REPO}"

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------- Header
md("# Advanced Machine Learning and MLOps\n"
   "## Project: Tourism Package Prediction")

md("# Problem Statement")

md("## **Business Context**")

md('"Visit with Us," a leading travel company, is revolutionizing the tourism industry by '
   "leveraging data-driven strategies to optimize operations and customer engagement. While "
   "introducing a new package offering, such as the Wellness Tourism Package, the company faces "
   "challenges in targeting the right customers efficiently. The manual approach to identifying "
   "potential customers is inconsistent, time-consuming, and prone to errors, leading to missed "
   "opportunities and suboptimal campaign performance.\n\n"
   "To address these issues, the company aims to implement a scalable and automated system that "
   "integrates customer data, predicts potential buyers, and enhances decision-making for "
   "marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless "
   "integration of data preprocessing, model development, deployment, and CI/CD practices for "
   "continuous improvement. This system will ensure efficient targeting of customers, timely "
   "updates to the predictive model, and adaptation to evolving customer behaviors, ultimately "
   "driving growth and customer satisfaction.")

md("## **Objective**")

md("As an MLOps Engineer at \"Visit with Us,\" your responsibility is to design and deploy an "
   "MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer "
   "purchases. The primary objective is to build a model that predicts whether a customer will "
   "purchase the newly introduced Wellness Tourism Package before contacting them. The pipeline "
   "will include data cleaning, preprocessing, transformation, model building, training, "
   "evaluation, and deployment, ensuring consistent performance and scalability. By leveraging "
   "GitHub Actions for CI/CD integration, the system will enable automated updates, streamline "
   "model deployment, and improve operational efficiency.")

md("## **Data Description**")

md("The dataset contains customer and interaction data that serve as key attributes for "
   "predicting the likelihood of purchasing the Wellness Tourism Package. The detailed "
   "attributes are:\n\n"
   "**Customer Details**\n"
   "- **CustomerID:** Unique identifier for each customer.\n"
   "- **ProdTaken:** Target variable indicating whether the customer has purchased a package "
   "(0: No, 1: Yes).\n"
   "- **Age:** Age of the customer.\n"
   "- **TypeofContact:** The method by which the customer was contacted (Company Invited or "
   "Self Inquiry).\n"
   "- **CityTier:** The city category based on development, population, and living standards "
   "(Tier 1 > Tier 2 > Tier 3).\n"
   "- **Occupation:** Customer's occupation (e.g., Salaried, Freelancer).\n"
   "- **Gender:** Gender of the customer (Male, Female).\n"
   "- **NumberOfPersonVisiting:** Total number of people accompanying the customer on the trip.\n"
   "- **PreferredPropertyStar:** Preferred hotel rating by the customer.\n"
   "- **MaritalStatus:** Marital status of the customer (Single, Married, Divorced).\n"
   "- **NumberOfTrips:** Average number of trips the customer takes annually.\n"
   "- **Passport:** Whether the customer holds a valid passport (0: No, 1: Yes).\n"
   "- **OwnCar:** Whether the customer owns a car (0: No, 1: Yes).\n"
   "- **NumberOfChildrenVisiting:** Number of children below age 5 accompanying the customer.\n"
   "- **Designation:** Customer's designation in their current organization.\n"
   "- **MonthlyIncome:** Gross monthly income of the customer.\n\n"
   "**Customer Interaction Data**\n"
   "- **PitchSatisfactionScore:** Score indicating the customer's satisfaction with the sales "
   "pitch.\n"
   "- **ProductPitched:** The type of product pitched to the customer.\n"
   "- **NumberOfFollowups:** Total number of follow-ups by the salesperson after the sales "
   "pitch.\n"
   "- **DurationOfPitch:** Duration of the sales pitch delivered to the customer.")

md("## Pre-requisites")

md("**This notebook runs locally (Jupyter), not in Google Colab.** The MLOps pipeline is "
   "deployed the same way the template describes, but the plumbing is adapted for a local "
   "environment:\n\n"
   "1. **GitHub repository:** [github.com/niteshtripa/tourism-package-prediction]"
   "(https://github.com/niteshtripa/tourism-package-prediction) — public repo created for this "
   "project.\n"
   "2. **GitHub Personal Access Token (classic)** with `repo` and `workflow` scopes, created at "
   "GitHub -> Settings -> Developer settings -> Personal access tokens. Instead of a Colab "
   "secret, it is read from the `GH_TOKEN` environment variable in this local session (never "
   "hard-coded in the notebook, and never printed to output).\n"
   "3. **GitHub Actions** runs the ML pipeline (data registration -> data prep -> model "
   "training/tracking -> commit model back to the repo) automatically on every push to `main`.\n"
   "4. **Streamlit Community Cloud** (share.streamlit.io) hosts the live app, deployed from "
   "`tourism_project/deployment/app.py` in the same repo.\n"
   "5. **MLflow** tracks experiments locally (SQLite backend) during development in this "
   "notebook; no ngrok/Colab tunnel is required outside Colab.")

md("## Installing and Importing Necessary Libraries")

code("!pip install -q mlflow==3.0.1 xgboost==2.1.1")

code("import os\n"
     "import subprocess\n"
     "\n"
     "import pandas as pd\n"
     "from sklearn.model_selection import train_test_split\n"
     "from sklearn.preprocessing import OneHotEncoder\n"
     "from sklearn.impute import SimpleImputer\n"
     "from sklearn.compose import make_column_transformer\n"
     "from sklearn.pipeline import make_pipeline\n"
     "import xgboost as xgb\n"
     "from sklearn.model_selection import GridSearchCV\n"
     "from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score\n"
     "import joblib\n"
     "import mlflow")

md("## Configuration")

code('# Edit these two values, then run every cell top to bottom.\n'
     'GITHUB_USERNAME = "niteshtripa"\n'
     'REPO_NAME       = "tourism-package-prediction"\n'
     '\n'
     'REPO   = f"{GITHUB_USERNAME}/{REPO_NAME}"\n'
     'BRANCH = "main"\n'
     'REPO_URL = f"https://github.com/{REPO}"\n'
     'print("Target repository:", REPO_URL)')

md("### GitHub Token")

md("The GitHub PAT is read from the `GH_TOKEN` environment variable of this local session "
   "(set beforehand in the terminal used to launch Jupyter) rather than a Colab secret, since "
   "this notebook is executed locally. It is never printed or written to a file — every helper "
   "below redacts it from any command output before printing.")

code('GH_TOKEN = os.environ.get("GH_TOKEN")\n'
     'print("GH_TOKEN loaded:", bool(GH_TOKEN))')

# ---------------------------------------------------------------- Model building
md("# Model Building")

code('# Create a master folder to keep all files created when executing the below code cells\n'
     'os.makedirs("tourism_project", exist_ok=True)')

code('# Create a folder for storing the model building files\n'
     'os.makedirs("tourism_project/model_building", exist_ok=True)')

md("## Data Registration")

code('os.makedirs("tourism_project/data", exist_ok=True)')

md("`tourism.csv` has been placed in `tourism_project/data/`. The dataset is registered with "
   "`tourism_project/model_building/data_register.py`, which checks that the expected columns "
   "are present and prints a summary.")

with open("tourism_project/model_building/data_register.py", encoding="utf-8") as f:
    data_register_src = f.read()
code(f'%%writefile tourism_project/model_building/data_register.py\n{data_register_src}')

code('!python tourism_project/model_building/data_register.py')

md("## Data Preparation")

md("`tourism_project/model_building/prep.py` loads the dataset from the repository data "
   "folder, drops identifier columns that carry no predictive signal (`Unnamed: 0`, "
   "`CustomerID`), fixes two data-entry issues (`\"Fe Male\"` -> `\"Female\"`, `\"Unmarried\"` "
   "-> `\"Single\"`), and splits the data into training and testing sets, saving them locally "
   "as CSV files. The GitHub Actions workflow passes these split files to the next job as a "
   "workflow artifact.")

with open("tourism_project/model_building/prep.py", encoding="utf-8") as f:
    prep_src = f.read()
code(f'%%writefile tourism_project/model_building/prep.py\n{prep_src}')

code('!python tourism_project/model_building/prep.py')

md("## Model Training and Registration with Experimentation Tracking")

md("`tourism_project/model_building/train.py` loads the train/test splits, builds a "
   "preprocessing + XGBoost pipeline, tunes it with `GridSearchCV`, logs parameters and metrics "
   "to MLflow, evaluates the best model, and saves it to `tourism_project/deployment/` so the "
   "workflow can commit it into the repo. XGBoost was chosen because it handles the tabular, "
   "mixed numeric/categorical features well and the class imbalance (~80/20) is corrected with "
   "`scale_pos_weight`.")

with open("tourism_project/model_building/train.py", encoding="utf-8") as f:
    train_src = f.read()
code(f'%%writefile tourism_project/model_building/train.py\n{train_src}')

code('!python tourism_project/model_building/train.py')

md("### Inspecting the MLflow Experiment Tracking Locally")

code('mlflow.set_tracking_uri("sqlite:///mlflow.db")\n'
     'runs = mlflow.search_runs(experiment_names=["tourism-package-prediction"])\n'
     'runs[[c for c in runs.columns if c.startswith("metrics.") or c.startswith("params.")]]')

# ---------------------------------------------------------------- Deployment
md("# Deployment")

code('os.makedirs("tourism_project/deployment", exist_ok=True)')

md("## Streamlit App")

md("`tourism_project/deployment/app.py` loads the model committed to `tourism_project/"
   "deployment/model.joblib` by the pipeline, collects the customer/interaction details "
   "through a form into a single-row dataframe, and displays the purchase prediction with its "
   "probability. Streamlit Community Cloud runs this file directly from the repo.")

with open("tourism_project/deployment/app.py", encoding="utf-8") as f:
    app_src = f.read()
code(f'%%writefile tourism_project/deployment/app.py\n{app_src}')

md("## App Dependencies")

md("`tourism_project/deployment/requirements.txt` pins the packages Streamlit Community Cloud "
   "installs before launching the app.")

with open("tourism_project/deployment/requirements.txt", encoding="utf-8") as f:
    deploy_req_src = f.read()
code(f'%%writefile tourism_project/deployment/requirements.txt\n{deploy_req_src}')

md("`tourism_project/requirements.txt` pins the packages the GitHub Actions pipeline installs "
   "for the data registration, data preparation, and model training jobs.")

with open("tourism_project/requirements.txt", encoding="utf-8") as f:
    pipeline_req_src = f.read()
code(f'%%writefile tourism_project/requirements.txt\n{pipeline_req_src}')

# ---------------------------------------------------------------- GH Actions
md("# MLOps Pipeline with GitHub Actions Workflow")

md("`.github/workflows/pipeline.yml` defines three jobs that run on every push to `main` "
   "(and can also be triggered manually from the Actions tab):\n\n"
   "1. **register-dataset** — installs dependencies, runs `data_register.py`, uploads the "
   "validated CSV as the `registered-data` artifact.\n"
   "2. **data-prep** — downloads `registered-data`, runs `prep.py`, uploads the train/test "
   "splits as the `data-splits` artifact.\n"
   "3. **model-training** — downloads `data-splits`, runs `train.py` (tuning + MLflow "
   "tracking), uploads the trained model as an artifact, and commits `model.joblib` back to "
   "the repository so Streamlit always serves the latest model.")

code('os.makedirs(".github/workflows", exist_ok=True)')

with open(".github/workflows/pipeline.yml", encoding="utf-8") as f:
    pipeline_yml_src = f.read()
code(f'%%writefile .github/workflows/pipeline.yml\n{pipeline_yml_src}')

md("## GitHub Authentication and Push Files")

md("Instead of the PyGithub content-API approach shown in the template, this project is a real "
   "local git repository. The cell below stages, commits, and pushes any changes to "
   f"`{REPO_URL}` using the token loaded from `GH_TOKEN`. The token is redacted from all "
   "printed output. It is safe to re-run: if there are no changes, git reports a clean working "
   "tree and nothing is pushed.")

code('def run(cmd, redact=None):\n'
     '    printable = " ".join(cmd).replace(redact, "***") if redact else " ".join(cmd)\n'
     '    print("$", printable)\n'
     '    result = subprocess.run(cmd, capture_output=True, text=True)\n'
     '    print(result.stdout)\n'
     '    if result.returncode != 0:\n'
     '        stderr = result.stderr.replace(redact, "***") if redact else result.stderr\n'
     '        print(stderr)\n'
     '    return result\n'
     '\n'
     'assert GH_TOKEN, "Set the GH_TOKEN environment variable before running this cell."\n'
     '\n'
     'run(["git", "add", "-A"])\n'
     'status = run(["git", "status", "--porcelain"])\n'
     '\n'
     'if status.stdout.strip():\n'
     '    run(["git", "commit", "-m", "Update project files from notebook run"])\n'
     '    push_url = f"https://{GH_TOKEN}@github.com/{REPO}.git"\n'
     '    push_result = run(["git", "push", push_url, f"HEAD:{BRANCH}"], redact=GH_TOKEN)\n'
     '    if push_result.returncode == 0:\n'
     '        print("Pushed changes to", REPO_URL)\n'
     '    else:\n'
     '        print("Push failed — see the error above (e.g. pull remote changes first).")\n'
     'else:\n'
     '    print("Nothing to commit — working tree already matches the last push to", REPO_URL)')

md("### Verifying the Repository Structure")

code('import urllib.request\n'
     'import json as _json\n'
     '\n'
     'required = [\n'
     '    "tourism_project/requirements.txt",\n'
     '    "tourism_project/model_building/data_register.py",\n'
     '    "tourism_project/model_building/prep.py",\n'
     '    "tourism_project/model_building/train.py",\n'
     '    "tourism_project/data/tourism.csv",\n'
     '    "tourism_project/deployment/app.py",\n'
     '    "tourism_project/deployment/requirements.txt",\n'
     '    ".github/workflows/pipeline.yml",\n'
     ']\n'
     '\n'
     'print("Verifying repository structure on GitHub...\\n")\n'
     'all_ok = True\n'
     'for path in required:\n'
     '    req = urllib.request.Request(\n'
     '        f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}",\n'
     '        headers={"Authorization": f"token {GH_TOKEN}"},\n'
     '    )\n'
     '    try:\n'
     '        urllib.request.urlopen(req)\n'
     '        print("OK   ", path)\n'
     '    except urllib.error.HTTPError:\n'
     '        print("MISSING", path)\n'
     '        all_ok = False\n'
     '\n'
     'print("\\nAll required files present on GitHub." if all_ok else "\\nSome files are missing.")')

md("### Checking the Latest GitHub Actions Run")

code('req = urllib.request.Request(\n'
     '    f"https://api.github.com/repos/{REPO}/actions/runs?per_page=1",\n'
     '    headers={"Authorization": f"token {GH_TOKEN}"},\n'
     ')\n'
     'run_info = _json.loads(urllib.request.urlopen(req).read())["workflow_runs"][0]\n'
     'print("Run:", run_info["display_title"])\n'
     'print("Status:", run_info["status"], "| Conclusion:", run_info["conclusion"])\n'
     'print("URL:", run_info["html_url"])')

md("**Note:** Replace every `<add_code_here>` from the original template's sample workflow — "
   "already done above — with the install command (`pip install -r tourism_project/"
   "requirements.txt`) and the `python ...` command for each script, before pushing.")

md("# Deploy the App on Streamlit Community Cloud")

md("After the pipeline finishes, the trained model is committed to `tourism_project/"
   "deployment/`. Deploy the app as follows:\n\n"
   "1. Go to **https://share.streamlit.io** and sign in with the GitHub account that owns the "
   f"repository (`{GITHUB_USERNAME}`).\n"
   "2. Click **Create app**.\n"
   "3. Set:\n"
   f"   - **Repository:** `{REPO}`\n"
   "   - **Branch:** `main`\n"
   "   - **Main file path:** `tourism_project/deployment/app.py`\n"
   "4. Open **Advanced settings** and set **Python version** to **3.11**.\n"
   "5. Click **Deploy**.\n\n"
   "Streamlit installs the packages from `tourism_project/deployment/requirements.txt` and "
   "launches the app at a public `...streamlit.app` URL.")

# ---------------------------------------------------------------- Output Evaluation
import base64
import os as _os

STREAMLIT_APP_URL = "https://tourism-package-prediction-gqebjzdhhzrn33gpitu3mw.streamlit.app/"
SCREENSHOT_DIR = _os.path.join("docs", "screenshots")


def embed_image(filename, alt):
    path = _os.path.join(SCREENSHOT_DIR, filename)
    if _os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"![{alt}](data:image/png;base64,{b64})"
    return f"_(screenshot `{filename}` not found in `{SCREENSHOT_DIR}/` — add it and rebuild)_"


md("# Output Evaluation")

md("### GitHub\n\n"
   f"- **Repository:** {REPO_URL}\n"
   f"- **Actions workflow runs:** {REPO_URL}/actions\n\n"
   "**Repository folder structure:**\n\n"
   f"{embed_image('github_repo_structure.png', 'GitHub repository folder structure')}\n\n"
   "**Successful GitHub Actions workflow run:**\n\n"
   f"{embed_image('github_actions_run.png', 'Successful GitHub Actions workflow run')}")

code('print("Repository:", REPO_URL)\n'
     'print("Actions:", REPO_URL + "/actions")')

md("### Streamlit Community Cloud\n\n"
   f"- **App URL:** {STREAMLIT_APP_URL}\n\n"
   "**Deployed app making a live prediction:**\n\n"
   f"{embed_image('streamlit_app_prediction.png', 'Streamlit app showing a prediction')}")

code(f'STREAMLIT_APP_URL = "{STREAMLIT_APP_URL}"\n'
     'print("Streamlit app:", STREAMLIT_APP_URL)')

md('<font size=6 color="navyblue">Power Ahead!</font>\n\n---')

nb["cells"] = cells

with open("Tourism_Package_Prediction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written with", len(cells), "cells")
