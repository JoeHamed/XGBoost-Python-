# XGBoost Classification Model

- This project demonstrates how to build and evaluate a machine learning classification model using the XGBoost algorithm.
- The dataset is processed, split into training and test sets, and the model is evaluated using a confusion matrix, accuracy score, and k-fold cross-validation.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Code Walkthrough](#code-walkthrough)
5. [Results](#results)
6. [License](#license)

## Introduction

This project uses the XGBoost (Extreme Gradient Boosting) algorithm for classification tasks. The dataset is first preprocessed to handle imbalanced classes, then split into training and test sets. The model is trained and evaluated, and performance metrics are generated, including a confusion matrix and cross-validation accuracy.

## Requirements

To run this project, you will need to have Python 3.x installed along with the following libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```
### Dataset
- The dataset used in this project is assumed to be a CSV file (`Data.csv`). The features are stored in `X`, and the target variable is stored in `y`.

### Data Preprocessing:
- The target variable y has class labels `2` and `4` that are mapped to `0` and `1`, respectively.
## Code Walkthrough
**1. Import Libraries:**
- `numpy`, `matplotlib`, and `pandas` are imported for data handling and visualization.
- The `XGBClassifier` from the `xgboost` library is used for classification.
  
**2. Dataset Import and Preprocessing:**
- The dataset is loaded using `pd.read_csv('Data.csv')`.
- The features `X` and target `y` are extracted from the dataset.
- Class labels `2` and `4` are converted to `0` and `1`, respectively.
**3. Splitting the Dataset:**
- The dataset is split into training and test sets using train_test_split from scikit-learn.
**4. Model Training:**
- The XGBClassifier is used to train the model on the training set.
**5. Confusion Matrix and Accuracy:**
- The model's performance is evaluated using the confusion matrix and accuracy score.
**6. k-Fold Cross Validation:**
- The model's stability and performance are further validated using 10-fold cross-validation.
## Results
- The confusion matrix shows that the model performs well with an accuracy of `97.81%`.
- The k-fold cross-validation results show an average accuracy of `96.71%` with a standard deviation of `2.28%`.
####                   [[85  2]

####  [ 1 49]]

