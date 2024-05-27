import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold

# Function to create a random hypervector
def create_hypervector(dim):
    return np.random.choice([1, -1], size=(dim,))

# Dataset schema and bin ranges
schema = {
    "age": (29, 77),
    "sex": (0, 1),
    "cp": (0, 3),
    "trtbps": (94, 200),
    "chol": (126, 564),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalachh": (71, 202),
    "exng": (0, 1),
    "oldpeak": (0, 6.2),
    "slp": (0, 2),    # Adjusted to match the dataset
    "caa": (0, 4),
    "thall": (0, 3),
    "output": (0, 1)
}

# Generate value hypervectors
def generate_value_hypervectors(D, M):
    return {feature: [create_hypervector(D) for _ in range(M)] for feature in schema if feature != "output"}

# Generate feature hypervectors
def generate_feature_hypervectors(D):
    return {feature: create_hypervector(D) for feature in schema if feature != "output"}

# Function to discretize feature values
def discretize_feature(value, feature, M):
    min_val, max_val = schema[feature]
    bin_edges = np.linspace(min_val, max_val, M + 1, endpoint=True)
    bin_index = np.digitize([value], bin_edges)[0] - 1
    bin_index = min(max(bin_index, 0), M - 1)
    return bin_index

# Function to encode a sample into a hypervector
def encode_sample(sample, value_hypervectors, feature_hypervectors, M):
    D = len(next(iter(value_hypervectors[next(iter(schema.keys()))])))
    hypervector = np.zeros(D)
    for feature, value in sample.items():
        if feature == "output":
            continue
        bin_index = discretize_feature(value, feature, M)
        value_hv = value_hypervectors[feature][bin_index]
        feature_hv = feature_hypervectors[feature]
        hypervector += np.multiply(value_hv, feature_hv)
    return np.sign(hypervector)

# Training
def train_model(train_data, D, M):
    value_hypervectors = generate_value_hypervectors(D, M)
    feature_hypervectors = generate_feature_hypervectors(D)
    class_hypervectors = {0: np.zeros(D), 1: np.zeros(D)}
    counts = {0: 0, 1: 0}
    for _, sample in train_data.iterrows():
        sample_hv = encode_sample(sample, value_hypervectors, feature_hypervectors, M)
        output = sample["output"]
        class_hypervectors[output] += sample_hv
        counts[output] += 1
    for output in class_hypervectors:
        class_hypervectors[output] /= counts[output]
    return class_hypervectors, value_hypervectors, feature_hypervectors

# Prediction using Cosine Similarity
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def predict(sample_hv, class_hypervectors):
    similarities = {output: cosine_similarity(sample_hv, class_hv) for output, class_hv in class_hypervectors.items()}
    return max(similarities, key=similarities.get)

# Load the dataset
data = pd.read_csv('heart.csv')

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['output']))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
scaled_data['output'] = data['output']

# Split the data into training (90%) and test (10%) sets
train_data, test_data = train_test_split(scaled_data, test_size=0.1, random_state=42)

# Best parameters from grid search
best_D = 5000
best_M = 10

# Train the model with best parameters
class_hypervectors, value_hypervectors, feature_hypervectors = train_model(train_data, best_D, best_M)

# Evaluate the model on the test set
correct_predictions = 0
test_samples = []

for _, test_sample in test_data.iterrows():
    test_sample_hv = encode_sample(test_sample, value_hypervectors, feature_hypervectors, best_M)
    predicted_class = predict(test_sample_hv, class_hypervectors)
    actual_class = test_sample["output"]
    correct_predictions += (predicted_class == actual_class)
    test_samples.append((predicted_class, actual_class))

accuracy = correct_predictions / len(test_data)
print("Final Test Accuracy:", accuracy)

# Print predicted vs actual outcomes for the first few test samples
for i, (predicted, actual) in enumerate(test_samples[:5]):
    print(f"Sample {i+1}: Predicted class: {predicted}, Actual class: {actual}")

# Cross-validation with best parameters
kf = KFold(n_splits=5, random_state=42, shuffle=True)
accuracies = []

for train_index, test_index in kf.split(scaled_data):
    train_data = scaled_data.iloc[train_index]
    test_data = scaled_data.iloc[test_index]
    
    class_hypervectors, value_hypervectors, feature_hypervectors = train_model(train_data, best_D, best_M)
    
    correct_predictions = 0
    for _, test_sample in test_data.iterrows():
        test_sample_hv = encode_sample(test_sample, value_hypervectors, feature_hypervectors, best_M)
        predicted_class = predict(test_sample_hv, class_hypervectors)
        actual_class = test_sample["output"]
        correct_predictions += (predicted_class == actual_class)
    
    accuracy = correct_predictions / len(test_data)
    accuracies.append(accuracy)

print("Cross-validated accuracy with best parameters:", np.mean(accuracies))
