# ==============================
# app/app.py
# ==============================

import sys
import os

# Allow imports from project root (so `from src.predict import ...` works)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request
from src.predict import predict_all

app = Flask(__name__)

# Dropdown options (order matters for ordinal fields)
JOB_TITLES = [
    "AI Engineer", "Backend Developer", "Business Analyst", "Cloud Engineer",
    "Cybersecurity Analyst", "Data Analyst", "Data Scientist", "DevOps Engineer",
    "Frontend Developer", "Machine Learning Engineer", "Product Manager", "Software Engineer"
]
EDUCATION_LEVELS = ["High School", "Diploma", "Bachelor", "Master", "PhD"]
INDUSTRIES = [
    "Consulting", "Education", "Finance", "Government", "Healthcare",
    "Manufacturing", "Media", "Retail", "Technology", "Telecom"
]
COMPANY_SIZES = ["Startup", "Small", "Medium", "Large", "Enterprise"]
LOCATIONS = ["Australia", "Canada", "Germany", "India", "Netherlands", "Remote", "Singapore", "Sweden", "UK", "USA"]
REMOTE_WORK_OPTIONS = ["No", "Hybrid", "Yes"]


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        job_titles=JOB_TITLES,
        education_levels=EDUCATION_LEVELS,
        industries=INDUSTRIES,
        company_sizes=COMPANY_SIZES,
        locations=LOCATIONS,
        remote_work_options=REMOTE_WORK_OPTIONS
    )


@app.route("/predict", methods=["POST"])
def predict():
    form_data = {
        "job_title": request.form["job_title"],
        "experience_years": request.form["experience_years"],
        "education_level": request.form["education_level"],
        "skills_count": request.form["skills_count"],
        "industry": request.form["industry"],
        "company_size": request.form["company_size"],
        "location": request.form["location"],
        "remote_work": request.form["remote_work"],
        "certifications": request.form["certifications"],
    }

    results = predict_all(form_data)

    return render_template(
        "result.html",
        predicted_salary=results["predicted_salary"],
        predicted_category=results["predicted_category"],
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)