# Credit Risk Modeling

### Predicting Loan Default with Interpretable Machine Learning

An applied credit-risk modelling project using applicant-level financial, demographic, employment, housing, and external-credit information to predict the probability of loan default.

The project develops a **leakage-controlled, interpretable modelling workflow**, progressing from data understanding and exploratory analysis through preprocessing, Logistic Regression benchmarking, class-imbalance analysis, and economically motivated feature engineering.

---

## Project Overview

Credit-risk modelling is fundamentally a **probability and ranking problem under asymmetric costs**. A useful model should not only distinguish between defaulting and non-defaulting borrowers, but should produce risk estimates that can support decisions such as:

* applicant risk ranking;
* credit approval;
* portfolio segmentation;
* further risk assessment;
* threshold-based lending decisions.

This project focuses on establishing a rigorous and interpretable modelling foundation before introducing more complex algorithms.

---

## Objective

The primary objective is to estimate the probability that a loan applicant will default based on information available at the time of application.

The modelling workflow specifically addresses:

* severe class imbalance;
* substantial missing data;
* mixed numerical and categorical variables;
* anomalous data representations;
* preprocessing leakage;
* probability-based risk ranking;
* threshold-dependent classification;
* economically motivated feature engineering.

---

## Dataset

The project uses the **Home Credit Default Risk** dataset.

The target variable is:

| `TARGET` | Interpretation |
| -------: | -------------- |
|        0 | No default     |
|        1 | Default        |

The application-level data contains information covering areas such as:

* income and employment;
* demographic characteristics;
* loan and payment characteristics;
* housing and assets;
* external credit information;
* credit-bureau enquiries;
* household characteristics.

The raw dataset is **not included in this repository**.

---

# Methodology

The project follows a sequential credit-risk modelling workflow.

```text
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Structural Data Preprocessing
        ↓
Leakage-Controlled ML Pipeline
        ↓
Logistic Regression Baseline
        ↓
Class-Imbalance Analysis
        ↓
Economic Feature Engineering
        ↓
Out-of-Sample Evaluation
        ↓
Advanced Modelling & Validation
```

---

## 01 — Data Understanding

The first stage establishes the structure and modelling context of the application dataset.

Key activities include:

* identifying the prediction target;
* separating identifiers from predictive variables;
* examining data types and variable structure;
* understanding economically meaningful feature groups;
* assessing the suitability of variables for credit-risk modelling.

---

## 02 — Exploratory Data Analysis

EDA is used to investigate the statistical and economic characteristics of the applicant population.

Areas examined include:

* target-class imbalance;
* missingness;
* numerical distributions;
* categorical characteristics;
* borrower characteristics associated with default;
* potentially anomalous variables.

The objective is to generate modelling hypotheses rather than perform visualization for its own sake.

---

## 03 — Data Preprocessing

Structural preprocessing addresses data-quality issues before model estimation.

The workflow includes:

* duplicate checks;
* missingness assessment;
* removal of variables with excessive missingness;
* examination of remaining missing values;
* preservation of potentially informative variables.

After structural cleaning, the modelling dataset contains **105 columns**.

---

## 04 — Machine Learning Pipeline

A leakage-controlled preprocessing architecture is implemented using scikit-learn's `ColumnTransformer` and `Pipeline`.

### Numerical features

```text
Median Imputation
       ↓
Standardization
```

### Categorical features

```text
Most-Frequent Imputation
       ↓
One-Hot Encoding
```

Categorical encoding uses:

```python
handle_unknown="ignore"
```

Preprocessing parameters are learned from the training sample and then applied to held-out observations.

This prevents information from the test set from influencing the transformation process.

---

# 05 — Baseline Credit Risk Model

## Logistic Regression

Logistic Regression is used as the primary interpretable baseline.

The model estimates:

$$
P(Y=1 \mid X)
$$

where \(Y=1\) represents default.

Logistic Regression provides a useful benchmark because it combines:

* probabilistic predictions;
* transparent coefficients;
* straightforward interpretation;
* computational efficiency;
* suitability as a benchmark for more complex models.

---

## Class Imbalance

Default is a minority outcome in the dataset.

Consequently, **accuracy is not used as the primary model-selection criterion**.

The project evaluates:

* ROC-AUC;
* PR-AUC;
* precision;
* recall;
* F1-score;
* threshold-dependent performance.

A class-weighted Logistic Regression specification is also evaluated to examine the effect of assigning greater importance to default observations.

An important distinction is maintained between:

> **ranking performance** and **classification performance at a particular threshold**.

Class weighting can alter the latter without necessarily improving the former.

---

# 06 — Feature Engineering

The feature-engineering stage introduces economically motivated variables intended to represent borrower characteristics and financial burden more directly.

Examples include transformations relating to:

* borrower age;
* employment;
* financial burden;
* loan characteristics.

The features are constructed exclusively from application-time information and do not use the target variable.

---

## Employment Anomaly

The variable `DAYS_EMPLOYED` contains a special coded value of `365243` for a substantial subset of observations.

Interpreting this value literally would imply an implausible employment duration.

The modelling treatment therefore:

1. identifies the special code;
2. treats it as missing when calculating employment duration;
3. creates an `EMPLOYMENT_ANOMALY` indicator;
4. retains the indicator as a potential predictive signal.

This preserves information contained in the original data representation while avoiding an economically nonsensical interpretation.

The association between the anomaly and default is treated as **predictive rather than causal**.

---

# Results

The initial modelling experiments establish the following out-of-sample results:

| Model                                    |    ROC-AUC |     PR-AUC |
| ---------------------------------------- | ---------: | ---------: |
| Logistic Regression — Baseline           |     0.7482 |     0.2283 |
| Logistic Regression — Feature Engineered | **0.7492** | **0.2313** |

The feature-engineered specification produces a modest improvement in both ROC-AUC and PR-AUC.

### Interpretation

The relatively small improvement is itself informative.

It suggests that much of the predictive information captured by the engineered variables is already represented in the original application features. At the same time, the engineered variables provide more economically interpretable representations of borrower characteristics and financial burden.

Therefore, feature engineering is evaluated based on **incremental predictive value and interpretability**, rather than assuming that additional variables will necessarily produce large performance gains.

---

# Credit-Risk Interpretation

Several practical lessons emerge from the modelling process.

### Accuracy is insufficient

With an imbalanced default outcome, a model can achieve high accuracy while providing poor identification of risky borrowers.

### Ranking matters

Credit-risk models are often used to rank applicants by estimated risk. ROC-AUC and PR-AUC therefore provide important information beyond a single classification threshold.

### Thresholds have economic consequences

A 0.50 probability threshold is a modelling convention, not necessarily an economically optimal lending cutoff.

The appropriate threshold depends on the relative costs of:

* approving a borrower who subsequently defaults;
* rejecting a borrower who would have repaid.

### Interpretability matters

Logistic Regression coefficients provide a transparent way to examine the direction and relative magnitude of model associations.

These coefficients should be interpreted as **conditional predictive associations, not causal effects**.

### More features do not automatically mean better models

The modest gain from feature engineering illustrates the importance of measuring incremental predictive value rather than assuming that additional transformations will materially improve discrimination.

---

# Repository Structure

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

The repository is intentionally organised around the analytical workflow, with each notebook representing a distinct stage of the modelling process.

---

# Reproducibility

The project uses:

* Python
* NumPy
* pandas
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter

Package versions are pinned in `requirements.txt`.

The raw Home Credit dataset is not included in the repository.

To reproduce the analysis:

1. Clone the repository.
2. Install the dependencies from `requirements.txt`.
3. Obtain the Home Credit Default Risk dataset separately.
4. Place the required data files in the expected local data directory.
5. Execute the notebooks sequentially from `01` through `06`.

---

# Limitations & Future Work

The current work establishes an interpretable baseline framework rather than a production lending model.

The next modelling stage will focus on:

* nonlinear benchmark models;
* cross-validation;
* model comparison;
* hyperparameter tuning;
* probability calibration;
* threshold optimisation;
* model stability;
* final model selection;
* business-cost analysis.

Further production-oriented extensions could include:

* out-of-time validation;
* population stability monitoring;
* model drift analysis;
* explainability;
* model governance documentation;
* champion/challenger frameworks.

A production credit-risk implementation would additionally require rigorous validation, monitoring, governance and regulatory review.

---

# Project Status

**Completed**

* Data understanding
* Exploratory data analysis
* Structural preprocessing
* Leakage-controlled ML pipeline
* Logistic Regression baseline
* Class-imbalance analysis
* Economic feature engineering
* Initial out-of-sample evaluation

**Next**

Advanced model comparison, validation, calibration, threshold analysis and final model selection.

---

## Author

**Piyush Sharma**
M.A. Economics — Delhi School of Economics

Applied credit-risk modelling project combining economic reasoning, statistical analysis and machine-learning methodology.
