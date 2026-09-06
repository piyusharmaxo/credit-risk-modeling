# Credit Risk Modeling

### Predicting Loan Default with Interpretable Machine Learning

An end-to-end credit-risk modeling project focused on predicting loan default, evaluating model discrimination and probability quality, and translating predicted risk into portfolio-level credit-risk segmentation and decision analysis.

The project follows a structured credit-risk workflow covering data understanding, exploratory analysis, structural preprocessing, machine-learning pipelines, baseline modeling, feature engineering, advanced model development, threshold optimization, explainability, and risk segmentation.

---

## Project Overview

Credit-risk modeling is a core application of statistical and machine-learning methods in financial risk management.

The objective of this project is to develop an applicant-level probability-of-default model using historical loan application data and evaluate how effectively the resulting model can:

- distinguish defaulting from non-defaulting applicants;
- estimate relative default risk;
- support risk-based portfolio segmentation;
- identify important predictive characteristics;
- evaluate alternative credit-decision thresholds;
- incorporate asymmetric costs of false positives and false negatives.

The project emphasizes methodological discipline and out-of-sample evaluation rather than maximizing a single predictive metric.

---

## Objective

Develop and evaluate a binary credit-default prediction model that:

1. establishes an interpretable Logistic Regression benchmark;
2. applies reproducible preprocessing and feature-engineering pipelines;
3. compares multiple model families using validation data;
4. selects and optimizes a final predictive model without using the test set for model selection;
5. evaluates discrimination, probability quality, and ranking performance;
6. analyzes threshold-dependent credit decisions under alternative cost assumptions;
7. provides model explainability and portfolio-level risk segmentation;
8. documents limitations and governance considerations relevant to credit-risk applications.

---

## Dataset

The project uses the **Home Credit Default Risk** dataset, an applicant-level historical loan application dataset containing financial, demographic, employment, housing, loan, and external credit-quality information.

### Target

`TARGET`

- `0` — applicant did not default
- `1` — applicant defaulted

### Identifier

`SK_ID_CURR` is retained for record identification but excluded from the machine-learning feature matrix.

### Major variable groups

- Demographic characteristics
- Income and employment
- Loan and payment information
- Housing and asset characteristics
- External credit-quality indicators
- Credit enquiry information
- Household and social characteristics
- Application-process variables

The raw dataset is intentionally excluded from version control.

---

## Methodology

The project follows the workflow:

**Data Understanding → EDA → Structural Preprocessing → ML Pipeline → Baseline Modeling → Feature Engineering → Advanced Model Development → Decision Analysis → Explainability → Risk Segmentation → Model Governance**

### Data preparation

- Dataset structure and variable types examined
- Target distribution assessed
- Missingness investigated
- Duplicate observations checked
- Structurally sparse variables removed using a >60% missingness threshold
- 105 columns retained after structural cleaning
- Imputation and categorical encoding deferred to the modeling pipeline

### Machine-learning pipeline

A reproducible preprocessing pipeline is implemented using:

- Median imputation for numerical variables
- Most-frequent imputation for categorical variables
- Standardization of numerical variables
- One-hot encoding of categorical variables
- `ColumnTransformer`
- Scikit-learn `Pipeline`

An 80/20 stratified train-test split with a fixed random seed is used for reproducibility.

---

## Project Stages

### 01 — Data Understanding

`notebooks/01_Data_Understanding.ipynb`

Establishes the analytical structure of the dataset, including:

- target definition;
- identifier treatment;
- variable types;
- missingness;
- major feature groups;
- initial data-quality assessment.

---

### 02 — Exploratory Data Analysis

`notebooks/02_EDA.ipynb`

Investigates:

- class imbalance;
- distributions of major variables;
- missingness patterns;
- applicant characteristics;
- relationships between financial characteristics and default;
- relevant bivariate patterns.

The EDA establishes the empirical context for subsequent modeling decisions.

---

### 03 — Data Preprocessing

`notebooks/03_Data_Preprocessing.ipynb`

Performs structural data cleaning while avoiding premature statistical imputation.

Key steps include:

- duplicate checks;
- structural missingness filtering;
- preservation of potentially informative variables;
- creation of the structurally cleaned dataset.

---

### 04 — Machine Learning Pipeline

`notebooks/04_Machine_Learning_Pipeline.ipynb`

Builds the reusable preprocessing framework for machine learning.

The pipeline separates numerical and categorical transformations and ensures that preprocessing operations are fitted only on training data.

---

### 05 — Baseline Credit Risk Modeling

`notebooks/05_Baseline_Credit_Risk_Modeling.ipynb`

Develops an interpretable Logistic Regression benchmark.

Evaluation includes:

- ROC-AUC;
- Precision-Recall AUC;
- precision;
- recall;
- F1 score;
- class imbalance considerations;
- threshold analysis;
- class-weighted Logistic Regression.

The 0.50 classification threshold is treated as an analytical reference point rather than an operational credit-policy threshold.

#### Baseline performance

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.7482 | 0.2283 |

---

### 06 — Feature Engineering

`notebooks/06_Feature_Engineering.ipynb`

Introduces economically motivated applicant-level features:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `LOAN_GOODS_RATIO`
- `AGE_YEARS`
- `EMPLOYMENT_YEARS`
- `EMPLOYMENT_ANOMALY`

The analysis also explicitly handles the anomalous `DAYS_EMPLOYED == 365243` value rather than treating it as a genuine employment duration.

#### Feature-engineered benchmark

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Feature-engineered Logistic Regression | 0.7492 | 0.2313 |

The improvement is modest but demonstrates the value of economically motivated feature construction.

---

### 07 — Advanced Credit Risk Model Development

`notebooks/07_Advanced_Credit_Risk_Model_Development.ipynb`

Develops and evaluates the final predictive model using a validation-based model-selection framework.

Model families considered include:

- Logistic Regression
- Random Forest
- Histogram-based Gradient Boosting

The final Gradient Boosting specification is selected using validation performance, with **PR-AUC as the primary model-selection metric** because default is a relatively rare outcome.

Hyperparameter optimization is performed using validation data.

The held-out test set remains untouched during model selection and threshold selection and is used only for final out-of-sample assessment.

---

## Final Model

### Histogram-based Gradient Boosting

The final model is a Histogram-based Gradient Boosting classifier.

### Final test-set performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.7603** |
| PR-AUC | **0.2495** |
| Brier Score | **0.067563** |
| KS Statistic | **0.3880** |

The final model improves upon the feature-engineered Logistic Regression benchmark on both ROC-AUC and PR-AUC.

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Baseline Logistic Regression | 0.7482 | 0.2283 |
| Feature-engineered Logistic Regression | 0.7492 | 0.2313 |
| **Final Histogram Gradient Boosting** | **0.7603** | **0.2495** |

This progression demonstrates the incremental development from an interpretable baseline toward a more flexible nonlinear model.

---

## Credit Decision Threshold Analysis

The model produces a continuous probability of default. Converting this probability into a binary credit-risk flag requires a decision threshold.

The project explicitly separates:

**Predictive model → estimated probability of default**

from

**Credit policy → decision threshold**

Threshold selection is therefore performed using validation data rather than the final test set.

### Validation-derived F1 threshold

The validation F1-optimal threshold is:

**0.16**

When evaluated on the untouched test set:

| Metric | Result |
|---|---:|
| Precision | 25.57% |
| Recall | 38.67% |
| F1 | 30.79% |
| Applicants flagged | 12.21% |

The threshold is not presented as a universal lending cutoff. An operational threshold would depend on the institution's risk appetite, underwriting strategy, operational capacity, and loss function.

---

## Cost-Sensitive Decision Analysis

A credit-risk system does not necessarily treat false positives and false negatives as equally costly.

- **False negative:** a borrower who defaults is not identified as high risk.
- **False positive:** a non-defaulting borrower is unnecessarily flagged.

Because institution-specific monetary loss estimates are unavailable, the project evaluates relative cost assumptions instead of assigning arbitrary monetary values.

The analysis demonstrates that increasing the relative cost of missed defaults leads to:

- lower operating thresholds;
- higher recall;
- more applicants being flagged;
- lower precision.

This reinforces the distinction between **predictive modeling** and **credit-policy design**.

---

## Model Explainability

The final Gradient Boosting model is analyzed using model-agnostic explainability techniques.

### Permutation Importance

Permutation importance is used to identify predictors that contribute most to out-of-sample discrimination.

PR-AUC is used as the importance metric because it is the primary model-selection metric for the imbalanced default-prediction problem.

The analysis identifies the external credit-quality variables as dominant predictive features, while several engineered variables also contribute meaningful predictive information.

Importance is interpreted as **predictive contribution rather than causal effect**.

### Robust Importance

Permutation importance is repeated across multiple random permutations to assess the stability of feature-importance estimates.

Mean importance and standard deviation are examined rather than relying on a single permutation.

### Partial Dependence

Partial dependence analysis is used to examine directional model relationships for selected continuous predictors.

Examples include:

- an inverse relationship between `EXT_SOURCE_2` and model-estimated default probability;
- a positive nonlinear relationship between `LOAN_GOODS_RATIO` and predicted default probability.

These relationships describe the behavior of the fitted model and should not be interpreted as causal effects.

---

## Portfolio Risk Segmentation

The final model's predicted probabilities are used to construct validation-derived risk bands.

Ten risk-band boundaries are estimated from the validation sample and then applied unchanged to the test set.

For each risk segment, the analysis evaluates:

- applicant population;
- average predicted probability;
- observed default rate;
- default lift;
- proportion of observed defaults captured.

### Three-band analytical segmentation

For portfolio interpretation, the resulting risk distribution is summarized into Low, Medium, and High Risk groups.

| Risk Segment | Population Share | Observed Default Rate | Portfolio Lift | Default Capture |
|---|---:|---:|---:|---:|
| Low Risk | ~50.2% | 2.83% | — | — |
| Medium Risk | ~30.5% | 8.21% | — | — |
| **High Risk** | **19.39%** | **21.41%** | **2.65×** | **51.44%**|

The analytical High Risk segment contains approximately one-fifth of applicants while capturing more than half of observed defaults.
This demonstrates meaningful portfolio-level risk stratification.

The segmentation is an analytical evaluation framework and should not be interpreted as a regulatory or institution-specific credit-grade system.

---

## Kolmogorov–Smirnov Analysis

The Kolmogorov–Smirnov statistic is included as an additional ranking diagnostic commonly used in credit-scoring applications.

The final model achieves:

**KS = 0.3880**

on the held-out test set.

The maximum separation occurs at approximately 35.15% of the ranked applicant population.

KS is treated as a discrimination diagnostic and is not used to determine the operational credit-decision threshold.

---

## Probability Quality

Discrimination metrics alone do not establish whether predicted probabilities correspond to observed default frequencies.

The project therefore evaluates probability quality separately using:

- Brier Score;
- calibration analysis.

The final test-set Brier Score is:

**0.067563**

These diagnostics are used for final model assessment and do not inform model selection.

---

## Model Governance and Limitations

The model is intended as a research and analytical credit-risk model rather than a production-ready lending decision system.

Important limitations include:

### Data and population limitations

Performance reflects the historical applicant population and lending process represented in the development dataset. Generalization to materially different populations, products, economic environments, or underwriting policies is not guaranteed.

### Selection considerations

Observed default outcomes reflect historical lending and application processes. Selection effects may therefore influence the modeling population.

### Feature limitations

External credit-quality variables contain substantial predictive information, but their underlying construction and data-generating processes are external to this project.

### Predictive versus causal interpretation

Model coefficients, feature importance, and partial-dependence relationships describe predictive associations and model behavior. They do not establish causal effects.

### Threshold and policy limitations

Predicted probability and credit-decision threshold are separate components of the risk-management framework.

An operational threshold should reflect:

- institutional risk appetite;
- expected loss;
- cost of missed defaults;
- cost of unnecessary interventions;
- operational capacity;
- underwriting policy.

### Fairness and governance

Demographic or potentially sensitive characteristics require additional governance and fairness review before operational deployment.

Predictive usefulness alone is not sufficient justification for including a feature in an actual lending decision.

### Monitoring

A production implementation would require ongoing monitoring of:

- ROC-AUC and PR-AUC;
- calibration;
- population stability;
- feature drift;
- missingness;
- default rates;
- segment-level performance.

### Production deployment

Before operational use, the model would require independent validation, documented data lineage, governance approval, fairness assessment, monitoring procedures, and institution-specific credit-policy design.

---

## Final Model Card

| Component | Final Specification |
|---|---|
| Problem | Binary credit-default prediction |
| Final Model | Histogram-based Gradient Boosting |
| Model Selection | Validation-based model comparison and hyperparameter optimization |
| Training Observations | 246,008 |
| Test Observations | 61,503 |
| Test Default Rate | 8.07% |
| Input Features | 109 |
| ROC-AUC | 0.7603 |
| PR-AUC | 0.2495 |
| Brier Score | 0.067563 |
| KS Statistic | 0.3880 |
| F1-optimal Validation Threshold | 0.16 |
| Test Precision at 0.16 | 25.57% |
| Test Recall at 0.16 | 38.67% |
| Test F1 at 0.16 | 30.79% |
| Test Flagged Rate at 0.16 | 12.21% |
| Highest-risk risk-band default rate | 21.41% |
| Highest-risk risk-band lift | 2.65× |
| High-risk Population Share | 19.39% |
| High-risk Default Capture | 51.44% |

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
    ├── 06_Feature_Engineering.ipynb
    └── 07_Advanced_Credit_Risk_Model_Development.ipynb