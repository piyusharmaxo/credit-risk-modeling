# Credit Risk Modeling

## Predicting Loan Default Using Explainable Machine Learning

An applied credit-risk modelling project focused on predicting loan default using applicant-level financial, demographic, employment, housing and external-credit information.

The project follows a structured machine-learning workflow with emphasis on:

* data quality and exploratory analysis;
* leakage-safe preprocessing;
* class-imbalance analysis;
* interpretable baseline modelling;
* economically motivated feature engineering;
* out-of-sample model evaluation.

---

## Objective

The objective is to develop an interpretable framework for estimating the probability of loan default using information available at the time of application.

The project treats credit-risk modelling as more than a generic binary-classification problem, with particular attention to:

* highly imbalanced default outcomes;
* substantial missing data;
* heterogeneous numerical and categorical variables;
* anomalous data codes;
* prevention of train-test leakage;
* probability-based risk ranking;
* economically meaningful feature construction.

---

## Dataset

The project uses the **Home Credit Default Risk** dataset.

The target variable is:

| Value | Meaning    |
| ----- | ---------- |
| `0`   | No default |
| `1`   | Default    |

The dataset contains applicant-level information covering areas such as:

* income and employment;
* demographic characteristics;
* loan and payment information;
* housing and assets;
* external credit information;
* credit-bureau enquiries;
* household characteristics.

The raw dataset is not included in this repository.

---

## Analytical Workflow

The project is organised into sequential notebooks:

### 01 — Data Understanding

Establishes the structure, variables, identifiers and economic context of the dataset.

### 02 — Exploratory Data Analysis

Investigates distributions, missingness, class imbalance and relationships between applicant characteristics and default.

### 03 — Data Preprocessing

Performs structural cleaning and prepares the modelling dataset while preserving economically relevant variables.

### 04 — Machine Learning Pipeline

Builds a leakage-controlled preprocessing pipeline for numerical and categorical variables using scikit-learn.

### 05 — Baseline Credit Risk Modeling

Establishes an interpretable Logistic Regression benchmark and evaluates performance using metrics appropriate for imbalanced classification.

### 06 — Feature Engineering

Introduces economically motivated borrower-level features and evaluates their incremental predictive value against the baseline.

---

## Methodological Principles

### Leakage Prevention

Preprocessing transformations are fitted using training data and subsequently applied to held-out data.

### Class Imbalance

Accuracy is not treated as the primary performance measure. ROC-AUC, PR-AUC and threshold-dependent metrics are considered together.

### Interpretability

Logistic Regression provides an interpretable probabilistic benchmark through model coefficients and estimated default probabilities.

### Economic Motivation

Feature engineering is guided by plausible credit-risk mechanisms rather than arbitrary transformations.

---

## Current Project Status

The initial analytical workflow has been completed through feature engineering.

The next stage focuses on:

* advanced model comparison;
* model validation;
* probability calibration;
* threshold analysis;
* final model selection;
* business and credit-risk interpretation.

---

## Repository Structure

```text
credit-risk-modeling/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── notebooks/
    ├── 01_Data_Understanding.ipynb
    ├── 02_EDA.ipynb
    ├── 03_Data_Preprocessing.ipynb
    ├── 04_Machine_Learning_Pipeline.ipynb
    ├── 05_Baseline_Credit_Risk_Modeling.ipynb
    └── 06_Feature_Engineering.ipynb
```

---

## Tools

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Author

**Piyush Sharma**
M.A. Economics — Delhi School of Economics
