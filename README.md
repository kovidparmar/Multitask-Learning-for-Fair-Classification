# Multitask Learning for Fair Classification

The implementation inspired by the paper:

> **“Taking Advantage of Multitask Learning for Fair Classification”**

The paper proposes that multitask learning can improve fairness by combining:

* shared learning across groups,
* and group-specific adaptations.

Paper Link:  
https://arxiv.org/pdf/1810.08683

This project explores fairness-aware machine learning using the **Adult Income Dataset** with **gender as the sensitive attribute**. The implementation compares:

* Single Task Learning (STL)
* Independent Task Learning (ITL)
* Multitask Learning (MTL)
* Fairness-Constrained MTL
* Predicted Sensitive Attribute MTL

The project focuses on:

* fairness-aware optimization,
* Equal Opportunity fairness,
* multitask learning,
* and fairness-performance tradeoffs.

---

# Project Motivation

Machine learning systems often produce uneven performance across demographic groups. Standard models optimized only for accuracy may unintentionally create unfair outcomes.

This project investigates whether:

* multitask learning can improve fairness,
* fairness constraints can reduce group disparity,
* and predicted sensitive attributes can support fairness-aware learning without directly using protected information at inference time.


---

# Key Features

* Adult Income Dataset experiments
* Gender-based fairness analysis
* STL vs ITL vs MTL comparison
* Equal Opportunity fairness constraints
* Predicted sensitive attribute pipeline
* Fairness vs accuracy tradeoff visualization


---

# Models Implemented

## 1. STL (Single Task Learning)

A single shared classifier trained on the entire dataset.

Implemented using:

```python
LogisticRegression
```

---

## 2. ITL (Independent Task Learning)

Separate classifiers trained independently for each sensitive group.

Implemented using:

```python
LogisticRegression
```

---

## 3. MTL (Multitask Learning)

A multitask linear classification framework implemented using:

* CVXPY optimization
* shared parameters
* group-specific parameters

The formulation used:

[
w_s = w_0 + v_s
]

where:

* (w_0) is the shared global model,
* (v_s) is the group-specific component,
* and (w_s) is the final classifier for group (s).

---

## 4. Fairness-Constrained MTL

MTL extended with:

* Equal Opportunity fairness constraints,
* fairness-aware optimization,
* and DEOd minimization.

---

## 5. Predicted Sensitive Attribute MTL

A privacy-aware pipeline where:

* gender is predicted using a RandomForest classifier,
* and the predicted gender is used for fairness-aware MTL.

Pipeline:

```text
Input Features (X)
        ↓
RandomForestClassifier
        ↓
Predicted Gender
        ↓
MTL Classifier
        ↓
Final Income Prediction
```

---

# Fairness Metric

This project uses:

## Difference in Equal Opportunity (DEOd)

DEOd measures disparity in true positive rates across sensitive groups.

* Lower DEOd = better fairness
* Higher DEOd = larger group disparity

This metric is particularly important in fairness-aware classification because it evaluates whether qualified individuals are treated similarly across groups.

---

# Methodology

The project workflow:

1. Load and preprocess Adult dataset
2. Perform one-hot encoding and feature scaling
3. Train STL and ITL baselines
4. Implement MTL using CVXPY
5. Add fairness constraints
6. Predict sensitive attributes
7. Evaluate:

   * accuracy,
   * fairness,
   * group-wise performance
8. Visualize fairness-performance tradeoffs

---

# Notebook Structure

## `01_preprocessing.ipynb`

* Dataset loading
* Cleaning
* One-hot encoding
* Feature scaling
* Train-test split
* Sensitive attribute extraction
* Dataset analysis and visualization

---

## `02_baselines.ipynb`

* STL implementation
* ITL implementation
* Baseline fairness analysis
* Group-wise accuracy evaluation

---

## `03_mtl.ipynb`

* MTL optimization
* Shared + group-specific learning
* CVXPY implementation
* MTL evaluation and visualization

---

## `04_fairness.ipynb`

* Equal Opportunity constraints
* Fairness-aware optimization
* Fairness vs accuracy tradeoff analysis

---

## `05_sensitive_prediction.ipynb`

* Sensitive attribute prediction
* RandomForest implementation
* Predicted-sensitive MTL pipeline
* Privacy-aware fairness evaluation

---

## `06_results.ipynb`

* Final comparison tables
* Combined visualizations
* Experimental conclusions
* Research discussion

---

# Experimental Results

## Main Results

| Model                     | Accuracy | DEOd   |
| ------------------------- | -------- | ------ |
| STL                       | 0.8460   | 0.0998 |
| ITL                       | 0.8454   | 0.1024 |
| MTL                       | 0.8423   | 0.0858 |
| MTL + Fairness Constraint | 0.8417   | 0.0240 |
| Predicted Sensitive MTL   | 0.8425   | 0.0222 |

---

## Group-wise Accuracy

| Model | Group 0 | Group 1 |
| ----- | ------- | ------- |
| STL   | 0.9176  | 0.8108  |
| ITL   | 0.9172  | 0.8101  |
| MTL   | 0.9166  | 0.8058  |

---

## Sensitive Attribute Prediction

| Task              | Accuracy |
| ----------------- | -------- |
| Gender Prediction | 0.8527   |

---

# Key Findings

## 1. STL and ITL show fairness disparity

Although STL and ITL achieved strong overall accuracy, both models produced noticeable group-wise disparity.

This demonstrates that:

* high accuracy alone does not guarantee fairness,
* and standard learning approaches may generalize unevenly across sensitive groups.

---

## 2. MTL improves fairness balance

MTL reduced DEOd compared to STL and ITL while maintaining competitive predictive performance.

This supports the paper’s hypothesis that:

* shared representations,
* combined with group-specific learning,
  can improve fairness-aware generalization.

---

## 3. Fairness constraints significantly reduce disparity

Adding Equal Opportunity constraints reduced DEOd from:

```text
0.0858 → 0.0240
```

while causing only a very small accuracy reduction.

This demonstrates an effective fairness-performance tradeoff.

---

## 4. Predicted sensitive attributes remain effective

Using predicted gender instead of true gender:

* preserved most of the predictive performance,
* and further improved fairness.

This is important because:

* sensitive attributes may not always be available,
* and privacy-aware fairness learning is increasingly important in real-world ML systems.

---

# Visualizations

The project includes:

* accuracy comparison plots,
* fairness comparison plots,
* group-wise accuracy plots,
* fairness vs accuracy scatter plots,
* and fairness tradeoff curves.

These visualizations help illustrate:

* fairness-performance tradeoffs,
* group disparities,
* and the effect of fairness-aware optimization.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd fair-multitask-learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run notebooks sequentially:

```text
01_preprocessing.ipynb
↓
02_baselines.ipynb
↓
03_mtl.ipynb
↓
04_fairness.ipynb
↓
05_sensitive_prediction.ipynb
↓
06_results.ipynb
```

---

# Project Structure

```text
.
├── 01_preprocessing.ipynb
├── 02_baselines.ipynb
├── 03_mtl.ipynb
├── 04_fairness.ipynb
├── 05_sensitive_prediction.ipynb
├── 06_results.ipynb
├── requirements.txt
│
├── data/
│   ├── processed/
│   └── results/
│
└── src/
    ├── config.py
    ├── data_utils.py
    ├── fairness.py
    ├── metrics.py
    ├── models.py
    ├── mtl.py
    ├── results_utils.py
    └── viz.py
```

---

# Future Work

Potential future improvements:

* Deep multitask learning architectures
* Multiple sensitive attributes
* Neural fairness-aware optimization
* Additional fairness datasets
* Calibration-aware fairness methods


