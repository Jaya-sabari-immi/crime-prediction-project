# MINI-PROJECT

**1. Prepare your environment**

Create a project folder and (optionally) a Python virtual environment:

mkdir crime-stacking
cd crime-stacking
python -m venv venv
source venv/bin/activate    # Linux / macOS
# .\venv\Scripts\activate   # Windows PowerShell


Install required packages:

pip install --upgrade pip
pip install pandas numpy matplotlib scikit-learn xgboost joblib

(Optional) Freeze dependencies:

pip freeze > requirements.txt


Tip: Use Python 3.8+ for best compatibility.

**2. Add code & dataset to the repo**

Place your main script (e.g., crime_prediction_stack.py) into the project root. If your code is inside Source code.pdf, copy it into a .py file.
Put the dataset balanced_crime_data.csv in the project root (or update the path in the script).
Ensure your repository contains:

crime_prediction_stack.py
balanced_crime_data.csv
README.md
requirements.txt

**3. Understand expected dataset format**

Your script expects:
A CSV with a target column named Arrest (binary; 0/1).
Columns that may include: ID, Case Number, Date, Updated On, Location (these are dropped by script if present).
Numerical and categorical features mixed; code auto-detects numeric vs non-numeric.
If your CSV differs: change the target variable and drop_cols list near the top of the script.

**4. Run the training script (first run)**

Execute:
python crime_prediction_stack.py

What happens:
Data loads and irrelevant columns are dropped.
Train/test split (80/20) with stratification.

Preprocessing:
numeric: median imputer + StandardScaler
categorical: most_frequent imputer + OneHotEncoder
Base models created: RandomForestClassifier & XGBClassifier.
Meta model: MLPClassifier wrapped in CalibratedClassifierCV.
Stacking pipeline is trained and saved.
Evaluations printed: classification report, accuracy, precision, recall, F1, ROC-AUC.
Precision–Recall vs Threshold plot shown.
Trained model saved to /mnt/data/saved_models/crime_stacking_model_v2.pkl.

**5. Verify model artifact**

After training, check:

ls -l /mnt/data/saved_models
# or, in repo: ls saved_models

Load model interactively:

import joblib
model = joblib.load("saved_models/crime_stacking_model_v2.pkl")
# Inspect attributes
print(model.named_steps.keys())   # should show 'preprocessor' and 'stacking'

**6. Evaluate & tune threshold (explainable)**

The script computes y_proba = model.predict_proba(X_test)[:, 1].
It finds a threshold that maximizes F1 via precision_recall_curve.
You can change metric to 'recall' or 'precision' when calling tune_threshold to prioritize other metrics.
To run tuning manually:
best_t, best_score = tune_threshold(y_test, y_proba, metric='f1')
print(best_t, best_score)

**7. Reproducibility & random seeds**

The script sets random_state=42 in train_test_split and models for reproducibility.
To reproduce results across machines, ensure same library versions (use requirements.txt).

**8. Improve or extend the pipeline (practical ideas)**

Feature engineering: add interaction features, date-time features, or aggregate features.
Imbalance handling: replace class_weight="balanced" with SMOTE or RandomUnderSampler in a pipeline step if you want synthetic oversampling.
Hyperparameter tuning: wrap the pipeline in GridSearchCV or RandomizedSearchCV (use cv=StratifiedKFold).
Add more base learners: e.g., LogisticRegression, CatBoost, or LightGBM.
Save training metrics & plots automatically to plots/ and reports/ directories.

**9. Quick code snippets you may add**

A — Loading a new sample and predicting:

import joblib
import pandas as pd

model = joblib.load("saved_models/crime_stacking_model_v2.pkl")
X_new = pd.read_csv("new_samples.csv")   # same columns as X used in training
probs = model.predict_proba(X_new)[:,1]
preds = (probs >= 0.5).astype(int)       # choose threshold

B — Persisting threshold & metadata:

import json
metadata = {"best_threshold": float(best_t), "model_name": "crime_stacking_model_v2.pkl"}
with open("saved_models/metadata.json","w") as f:
    json.dump(metadata, f, indent=2)

**10. Unit tests & sanity checks**

Add simple assertions to avoid silent failures:

assert 'Arrest' in df.columns, "Dataset must contain target column 'Arrest'"
assert df['Arrest'].nunique() == 2, "Target must be binary"

**11. Deployment options (brief)**

Batch inference: schedule the script to predict nightly on new data, store results in DB/CSV.
REST API: wrap model in a Flask/FastAPI app and serve predict endpoint. Example:
POST /predict with JSON rows → return probability and prediction.
Cloud: containerize with Docker and deploy to AWS ECS, GCP Cloud Run, or Azure App Service.

**12. Troubleshooting common issues**

XGBoost errors: If use_label_encoder=False triggers warnings, ensure xgboost is updated.
Memory issues: OneHotEncoding can blow up dimensionality — consider TargetEncoder or HashingEncoder for very high-cardinality columns.
Long training time: reduce n_estimators or train on a smaller sample while prototyping.
Different dataset path: update df = pd.read_csv("...") to correct path.

**13. Suggested repo additions (for professionalism)**

LICENSE (MIT or your choice)

CONTRIBUTING.md (how to contribute)

CODE_OF_CONDUCT.md (optional)

notebooks/ with EDA notebook & model explainability (SHAP/LIME)

examples/ folder with predict_example.csv and usage demo

**14. Checklist before submission / sharing**

 requirements.txt present and up to date
 Script runs end-to-end on a clean environment
 Saved model and metadata.json included (or generation steps documented)
 README.md contains run instructions and dataset expectations
 No hard-coded absolute paths (or clearly documented)
 Sensitive data or API keys are not committed
