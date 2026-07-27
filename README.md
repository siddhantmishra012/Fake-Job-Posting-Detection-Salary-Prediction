# Fake Job Posting Detection & Salary Prediction

## Academic Project Overview

This project implements a complete, end-to-end Machine Learning pipeline developed for a B.Tech Computer Science & Engineering (CSE) coursework assignment. The system addresses two core objectives in online recruitment security and compensation estimation:

1. **Salary Prediction (Regression):** Predicts an estimated candidate salary (in INR) based on parameters such as experience level, education degree, company size, and employment type.
2. **Fake Job Detection (Classification):** Classifies job postings as either Genuine (`0`) or Fake (`1`) by evaluating security indicators, company credibility markers, and recruitment tactics.

The implementation strictly uses standard Python libraries: NumPy, Pandas, Matplotlib, Scikit-Learn, and Pickle.

---

## Table of Contents
- [Project Objectives](#project-objectives)
- [Technologies Used](#technologies-used)
- [Dataset Specifications](#dataset-specifications)
- [Project Directory Structure](#project-directory-structure)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Execution Guide](#execution-guide)
- [Phase Summary & Outputs](#phase-summary--outputs)
- [Future Enhancements](#future-enhancements)

---

## Project Objectives

- **Exploratory Data Analysis:** Analyze job listing metadata to uncover pay scales, education requirements, and fraudulent posting patterns.
- **Data Preprocessing:** Handle missing data, remove duplicates, and encode categorical variables using Scikit-Learn tools.
- **Regression Modeling:** Train and evaluate a Linear Regression model to predict continuous salary values.
- **Classification Modeling:** Train and evaluate a Logistic Regression classifier to distinguish fake job listings from real ones.
- **Model Persistence:** Serialize trained models to `.pkl` files for instant inference without retraining.
- **Interactive Inference CLI:** Provide a command-line interface for real-time user predictions, trust scoring, and fraud risk assessment.

---

## Technologies Used

| Tool / Library | Role in Project |
| :--- | :--- |
| **Python 3.8+** | Primary programming language |
| **NumPy** | Array handling and numerical operations |
| **Pandas** | Tabular data loading, manipulation, and summary statistics |
| **Matplotlib** | Generating exploratory visualization charts |
| **Scikit-Learn** | Data preprocessing, label encoding, train-test splitting, and ML model training |
| **Pickle** | Model serialization and deserialization |

---

## Dataset Specifications

The dataset (`dataset/jobs.csv`) contains **2,020 job records** with **17 feature attributes**:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Job_Title` | String | Position title (e.g., Data Scientist, Software Engineer) |
| `Company` | String | Employer or organization name |
| `Location` | String | Work city / location |
| `Company_Size` | Categorical | Enterprise, Large, Medium, Small |
| `Experience` | Categorical | Entry Level, Mid Level, Senior, Executive, Internship |
| `Degree` | Categorical | Bachelor's, Master's, PhD, Diploma, High School |
| `Job_Type` | Categorical | Full-Time, Part-Time, Contract, Freelance, Internship |
| `Industry` | Categorical | IT, Finance, Healthcare, Education, Manufacturing, etc. |
| `Remote` | Binary (0/1) | Work arrangement (0 = Onsite, 1 = Remote) |
| `Skills` | String | Required technical skills list |
| `Salary` | Integer | Annual salary target in INR |
| `Has_Company_Website` | Binary (0/1) | Flag indicating presence of company website |
| `Has_Company_Logo` | Binary (0/1) | Flag indicating presence of company logo |
| `Email_Domain` | Categorical | Employer contact email domain |
| `Application_Fee` | Binary (0/1) | Flag indicating if an upfront fee is demanded |
| `Urgent_Hiring` | Binary (0/1) | Flag indicating urgent recruitment tactics |
| `Fake` | Binary (0/1) | Target classification label (0 = Genuine, 1 = Fake) |

---

## Project Directory Structure

```text
Fake_Job_Posting_Detector/
│
├── dataset/
│   └── jobs.csv                      # Raw dataset file
│
├── graphs/
│   ├── actual_vs_predicted_salary.png
│   ├── confusion_matrix.png
│   └── (10 visualization outputs saved during Phase 2)
│
├── phase1_analysis.py                # Phase 1: Data Analysis & Insights
├── phase2_visualization.py           # Phase 2: Matplotlib Visualizations
├── phase3_preprocessing.py           # Phase 3: Cleaning & Encoding
├── phase4_linear_regression.py       # Phase 4: Salary Regression Model
├── phase5_logistic_regression.py     # Phase 5: Fake Classifier Model
├── phase6_prediction.py              # Phase 6: Terminal Interactive CLI
│
├── salary_model.pkl                  # Serialized Linear Regression Model
├── fake_detector.pkl                 # Serialized Logistic Regression Model
├── label_encoders.pkl                # Serialized LabelEncoders
└── README.md                         # Project documentation
```

---

## Machine Learning Methodology

### 1. Linear Regression (Salary Prediction)
- **Model Type:** Multiple Linear Regression
- **Target Variable:** `Salary`
- **Equation:** $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$
- **Metrics Evaluated:** Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared ($R^2$) score.

### 2. Logistic Regression (Fake Job Detection)
- **Model Type:** Binary Logistic Regression with Sigmoid Activation
- **Target Variable:** `Fake` (0 = Genuine, 1 = Fake)
- **Equation:** $P(Y=1|X) = \frac{1}{1 + e^{-z}}$
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## Execution Guide

### Prerequisite Installation
Ensure required dependencies are installed:
```bash
pip install numpy pandas matplotlib scikit-learn
```

### Running Project Modules
Execute the phases in sequence from the project root directory:

```bash
# 1. Exploratory Data Analysis & Statistics
python phase1_analysis.py

# 2. Generate and Save Visualizations
python phase2_visualization.py

# 3. Data Cleaning, Label Encoding & Feature Matrix Creation
python phase3_preprocessing.py

# 4. Train Linear Regression (Salary Prediction)
python phase4_linear_regression.py

# 5. Train Logistic Regression (Fake Job Detection)
python phase5_logistic_regression.py

# 6. Launch Real-time Interactive Prediction CLI
python phase6_prediction.py
```

---

## Phase Summary & Outputs

| Phase Script | Primary Function | Output Description |
| :--- | :--- | :--- |
| **`phase1_analysis.py`** | Statistical Inspection | Prints dataset shape, null counts, duplicate counts, and salary/fake job breakdown reports. |
| **`phase2_visualization.py`** | Data Visualization | Generates 10 charts (histograms, bar plots, pie charts, scatter plots, box plots) saved into `graphs/`. |
| **`phase3_preprocessing.py`** | Data Preprocessing | Imputes missing values, encodes categories, splits data 80:20, and exports `label_encoders.pkl`. |
| **`phase4_linear_regression.py`** | Salary Prediction | Fits Linear Regression, prints error metrics (MAE, RMSE, R²), and exports `salary_model.pkl`. |
| **`phase5_logistic_regression.py`** | Fake Classification | Fits Logistic Regression (82.75% accuracy), prints evaluation matrix, and exports `fake_detector.pkl`. |
| **`phase6_prediction.py`** | Interactive CLI | Interactive terminal menu evaluating custom job inputs to output Salary estimates, Fake status, Trust Score, and Red Flag analysis. |

---

## Future Enhancements

- **Natural Language Processing (NLP):** Incorporate TF-IDF vectorization on job description text to detect semantic scam signals.
- **Ensemble Algorithms:** Evaluate non-linear models such as Random Forest, Gradient Boosting, or Support Vector Machines (SVM).
- **Web Interface:** Connect model endpoints to a web dashboard for interactive browser-based testing.
- **Real-world Benchmarks:** Test models on large-scale public datasets such as EMSCAD.
