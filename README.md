

---

# Heart Attack Prediction Using Hyperdimensional Computing (HDC)

## Overview

This project explores the use of Hyperdimensional Computing (HDC) to predict heart attacks. HDC is inspired by brain-like computing and leverages high-dimensional vector spaces for efficient and robust pattern recognition. Our model encodes patient data as hypervectors, trains to distinguish between heart attack and non-heart attack cases, and optimizes hyperparameters to achieve high predictive accuracy. The model can be deployed on smartwatches for real-time heart attack prediction.

## Dataset

The dataset used is the UCI Heart Disease dataset, containing 303 samples with the following features:
- `age`: Age of the patient (29-77)
- `sex`: Sex of the patient (0-1)
- `cp`: Chest pain type (0-3)
- `trtbps`: Resting blood pressure (94-200)
- `chol`: Serum cholesterol (126-564)
- `fbs`: Fasting blood sugar (0-1)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalachh`: Maximum heart rate achieved (71-202)
- `exng`: Exercise-induced angina (0-1)
- `oldpeak`: ST depression induced by exercise relative to rest (0-6.2)
- `slp`: Slope of the peak exercise ST segment (0-2)
- `caa`: Number of major vessels (0-4)
- `thall`: Thalassemia (0-3)
- `output`: Target variable indicating presence of heart disease (0-1)

## Code Explanation

The main script for training and testing the model is `test.py`. Below is an explanation of the key parts of the code.

### Importing Libraries and Loading Data

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('heart.csv')
```

### Preprocessing

Normalize the features and split the data into training and test sets.

```python
# Normalize features
features = data.columns[:-1]
data[features] = (data[features] - data[features].min()) / (data[features].max() - data[features].min())

# Split the data
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
```

### Hyperdimensional Computing Functions

#### Discretization and Hypervector Encoding

```python
D = 5000  # Dimensionality
M = 10    # Number of bins

# Discretize feature values
def discretize_feature(value, feature):
    min_val, max_val = schema[feature]
    bin_width = (max_val - min_val) / M
    return int((value - min_val) / bin_width)

# Generate random hypervectors
def generate_hypervectors():
    value_hypervectors = {feature: [np.random.choice([-1, 1], D) for _ in range(M)] for feature in schema.keys()}
    feature_hypervectors = {feature: np.random.choice([-1, 1], D) for feature in schema.keys()}
    return value_hypervectors, feature_hypervectors

# Encode a sample
def encode_sample(sample):
    hypervector = np.zeros(D)
    for feature in schema.keys():
        value = sample[feature]
        bin_index = discretize_feature(value, feature)
        value_hv = value_hypervectors[feature][bin_index]
        feature_hv = feature_hypervectors[feature]
        hypervector += value_hv * feature_hv
    return np.sign(hypervector)
```

#### Training and Prediction

```python
# Train the model
def train_model(train_data):
    class_hypervectors = {0: np.zeros(D), 1: np.zeros(D)}
    for _, sample in train_data.iterrows():
        sample_hv = encode_sample(sample)
        class_hypervectors[int(sample['output'])] += sample_hv
    for key in class_hypervectors:
        class_hypervectors[key] = np.sign(class_hypervectors[key])
    return class_hypervectors

# Predict the class of a sample
def predict(sample, class_hypervectors):
    sample_hv = encode_sample(sample)
    similarities = {cls: np.dot(sample_hv, class_hypervectors[cls]) for cls in class_hypervectors}
    return max(similarities, key=similarities.get)
```

### Main Execution

Train the model, predict the test set, and calculate accuracy.

```python
# Load schema
schema = {
    'age': (29, 77),
    'sex': (0, 1),
    'cp': (0, 3),
    'trtbps': (94, 200),
    'chol': (126, 564),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalachh': (71, 202),
    'exng': (0, 1),
    'oldpeak': (0, 6.2),
    'slp': (0, 2),
    'caa': (0, 4),
    'thall': (0, 3)
}

# Generate hypervectors
value_hypervectors, feature_hypervectors = generate_hypervectors()

# Train the model
class_hypervectors = train_model(train_data)

# Test the model
predictions = test_data.apply(lambda row: predict(row, class_hypervectors), axis=1)
accuracy = accuracy_score(test_data['output'], predictions)
print(f'Final Test Accuracy: {accuracy}')
```

## Results

- **Final Test Accuracy:** 74.19%
- **Cross-validated Accuracy:** 82.84%

### Sample Predictions

| Sample | Predicted Class | Actual Class |
|--------|-----------------|--------------|
| 1      | 0               | 0.0          |
| 2      | 0               | 0.0          |
| 3      | 1               | 1.0          |
| 4      | 0               | 0.0          |
| 5      | 1               | 1.0          |

## Hyperparameter Tuning

A grid search over dimensionality \(D\) and number of bins \(M\) identified \(D = 5000\) and \(M = 10\) as the best parameters, resulting in the highest accuracy.

## Potential Applications

The HDC model's low computational requirement makes it suitable for deployment on smartwatches and other wearable devices, enabling real-time heart attack prediction and timely medical interventions.

## References

1. A. Rahimi, H. Zhu, A. Amrou, and J. M. Rabaey, "Hyperdimensional Computing for Noninvasive Brain–Computer Interfaces: Blind and Cross-Subject Classification of EEG Error-Related Potentials," Proceedings of the IEEE, 2017.
2. UCI Machine Learning Repository: Heart Disease Data Set. Available at: [https://archive.ics.uci.edu/ml/datasets/heart+Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---
