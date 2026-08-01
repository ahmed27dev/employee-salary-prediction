# Employee Salary Prediction

An end-to-end machine learning project that predicts an employee's **salary** (regression) and **salary category** (classification) based on role, experience, education, and company attributes — deployed as a Flask web application.

Built as a first complete ML portfolio project, following a full pipeline: exploratory data analysis → preprocessing → model training → evaluation → deployment.

---

## Demo

Fill in a role's details — job title, experience, education, industry, company size, location, remote work type, skills, and certifications — and get back:

- A **predicted salary** (Linear Regression)
- A **salary category** — Low / Medium / High (Logistic Regression)

---

## Project Structure

```
employee-salary-prediction/
│
├── app/
│   ├── static/
│   │   ├── css/style.css
│   │   ├── images/
│   │   └── js/script.js
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── app.py                      # Flask routes
│   └── __init__.py
│
├── data/
│   ├── raw/salary.csv              # Original dataset (250,000 rows)
│   └── processed/
│       ├── feature_inventory.csv
│       └── coefficients.csv        # Linear Regression coefficient interpretation
│
├── models/
│   ├── preprocessor.pkl                  # Fitted preprocessing pipeline (regression)
│   ├── linear_regression_model.pkl       # Trained regression model
│   ├── preprocessor_classifier.pkl       # Fitted preprocessing pipeline (classification)
│   ├── logistic_regression_model.pkl     # Trained classification model
│   └── label_encoder.pkl                 # Encodes/decodes Low/Medium/High labels
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_Preprocessing.ipynb      # Encoding, scaling, pipeline construction
│   ├── 03_LinearRegression.ipynb   # Regression training & evaluation
│   └── 04_Classification.ipynb     # Classification training & evaluation
│
├── src/
│   ├── preprocess.py               # Shared preprocessing pipeline
│   ├── train_regression.py         # Regression training pipeline
│   ├── train_classifier.py         # Classification training pipeline
│   ├── predict.py                  # Inference logic used by the Flask app
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── utils.py
│   └── __init__.py
│
├── images/
├── requirements.txt
└── README.md
```

> `.venv/`, `__pycache__/`, and `.ipynb_checkpoints/` are excluded via `.gitignore` and are not part of the repository.

---

## Dataset

- **250,000 rows**, no missing values, no duplicates
- **9 features**: `job_title`, `experience_years`, `education_level`, `skills_count`, `industry`, `company_size`, `location`, `remote_work`, `certifications`
- **Target (regression)**: `salary` (continuous, ₹31,867 – ₹333,046)
- **Target (classification)**: `salary_category` — derived via quantile binning into `Low` / `Medium` / `High` (balanced, ~33.3% each)

---

## Preprocessing

Built with a single `ColumnTransformer` combining three strategies, chosen per feature based on its type:

| Type                        | Features                                             | Technique                                     |
| --------------------------- | ---------------------------------------------------- | --------------------------------------------- |
| Numerical                   | `experience_years`, `skills_count`, `certifications` | `StandardScaler`                              |
| Ordinal (has natural order) | `education_level`, `company_size`, `remote_work`     | `OrdinalEncoder` with explicit category order |
| Nominal (no order)          | `job_title`, `industry`, `location`                  | `OneHotEncoder`                               |

Fitted strictly on training data only (no leakage into the test set) and serialized with `joblib` for reuse in the Flask app.

---

## Models & Results

### Linear Regression (Salary Prediction)

| Metric | Value                            |
| ------ | -------------------------------- |
| MAE    | ₹6,154.61 (~4.2% of mean salary) |
| RMSE   | ₹7,987.50                        |
| R²     | 0.9541                           |

**Key insight:** Location and job title are the dominant salary predictors — geography alone can swing predicted salary by tens of thousands, far outweighing experience, education, or industry. Industry has almost no measurable effect (all coefficients under ±100), suggesting salary in this dataset is driven primarily by role and geography rather than sector.

### Logistic Regression (Salary Category Prediction)

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 90.61% |
| Precision (macro) | 0.9065 |
| Recall (macro)    | 0.9061 |
| F1-score (macro)  | 0.9063 |

| Class  | Precision | Recall | F1   |
| ------ | --------- | ------ | ---- |
| High   | 0.94      | 0.93   | 0.93 |
| Low    | 0.93      | 0.92   | 0.93 |
| Medium | 0.85      | 0.87   | 0.86 |

**Key insight:** High and Low earners are never confused with each other. Nearly all misclassifications involve the Medium category, which occupies a comparatively narrow salary band (₹127,849–₹159,802) squeezed between two much wider bands — an expected consequence of quantile-based binning on a continuous variable near a narrow middle range.

---

## Tech Stack

- **Language:** Python 3.14
- **ML:** scikit-learn (LinearRegression, LogisticRegression, ColumnTransformer, Pipeline)
- **Data handling:** pandas, numpy
- **Web app:** Flask
- **Model persistence:** joblib

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/ahmed27dev/employee-salary-prediction.git
cd employee-salary-prediction

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app/app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> Trained model artifacts (`models/*.pkl`) are included in this repository so the app runs immediately after cloning. They can be regenerated from scratch by running the notebooks in `notebooks/` in order (01 → 04), which will rebuild and re-save each `.pkl` file via `src/preprocess.py`, `src/train_regression.py`, and `src/train_classifier.py`.

---

## Methodology Notes

- **No data leakage:** all preprocessing (`fit`) happens strictly on training data; test data only ever goes through `.transform()`.
- **Category-derived target leakage was explicitly handled:** since `salary_category` is derived from `salary`, the raw `salary` column is excluded from the classification feature set — otherwise the classifier could trivially "cheat" using the source column instead of learning from role/experience/education signals.
- **Stratified splitting** was used for the classification task to preserve class balance across train and test sets.
- Both models are packaged as complete pipelines (`ColumnTransformer` + model), so a single `.transform()` → `.predict()` call handles all encoding and scaling consistently between training and inference.

---

## Author

Ahmed — B.Tech CSE
