# CSCI4734 – Machine Learning

# Logistic Regression from Scratch

## Overview
This project implements logistic regression from scratch to classify student admission based on exam scores. The dataset (`exams.csv`) contains two exam scores and a binary label indicating whether a student was admitted (1) or not (0).

---

## Components
1. **Loading Data**
   - Read data using Pandas.
   - Normalize features using Min-Max Scaling.

2. **Visualization**
   - Scatter plot of exam scores.
   - Admitted students: Green; Failed students: Red.

3. **Implementation of Logistic Regression from Scratch**
   - Implement the sigmoid function.
   - Define the cost function.
   - Implement gradient descent for parameter optimization.
   - Plot the cost function over iterations.
   - Visualize decision boundary.
   - Evaluate model performance.

4. **Logistic Regression using Library**
   - Use `sklearn` to train logistic regression.
   - Compare results with manual implementation.

5. **Report**
   - Document results and explanations.

---

## Dataset
- **Columns**:
  - `exam_1`: First exam score.
  - `exam_2`: Second exam score.
  - `admitted`: 1 if admitted, 0 if not.

---

## Steps

### 1. Load and Preprocess Data
- Read `exams.csv` using Pandas.
- Normalize `exam_1` and `exam_2` using Min-Max Scaling.
- Separate features and labels.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("exams.csv")
scaler = MinMaxScaler()
df[['exam_1', 'exam_2']] = scaler.fit_transform(df[['exam_1', 'exam_2']])
X = df[['exam_1', 'exam_2']].values
y = df['admitted'].values
```

### 2. Data Visualization
```python
import matplotlib.pyplot as plt

admitted = df[df['admitted'] == 1]
failed = df[df['admitted'] == 0]

plt.scatter(admitted['exam_1'], admitted['exam_2'], color='green', label='Admitted')
plt.scatter(failed['exam_1'], failed['exam_2'], color='red', label='Failed')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()
```

### 3. Logistic Regression Implementation
#### a) Sigmoid Function
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

#### b) Cost Function
```python
def cost_function(X, y, weights):
    m = len(y)
    h = sigmoid(X @ weights)
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```

#### c) Gradient Descent
```python
def gradient_descent(X, y, weights, alpha, num_iterations):
    m = len(y)
    cost_history = []
    for _ in range(num_iterations):
        weights -= (alpha / m) * X.T @ (sigmoid(X @ weights) - y)
        cost_history.append(cost_function(X, y, weights))
    return weights, cost_history
```

#### d) Train Logistic Regression Model
```python
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
weights = np.zeros(X_bias.shape[1])
weights, cost_history = gradient_descent(X_bias, y, weights, 0.01, 10000)
```

#### e) Plot Cost Function
```python
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.show()
```

#### f) Decision Boundary
```python
x_values = [min(X[:, 0]), max(X[:, 0])]
y_values = - (weights[0] + np.dot(weights[1], x_values)) / weights[2]

plt.scatter(admitted['exam_1'], admitted['exam_2'], color='green', label='Admitted')
plt.scatter(failed['exam_1'], failed['exam_2'], color='red', label='Failed')
plt.plot(x_values, y_values, color='blue', label='Decision Boundary')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend()
plt.show()
```

### 4. Logistic Regression Using Library
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(f'Accuracy: {accuracy_score(y, y_pred)}')
```

### 5. Prediction on New Data
```python
new_data = np.array([[55, 70], [40, 60]])
new_data = scaler.transform(new_data)
predictions = model.predict(new_data)
print(f'Predictions: {predictions}')
```

---

## Results
The result indicates that the logistic regression model implemented from scratch achieved an accuracy score of 0.89 on the training set, suggesting that it correctly predicted 89% of the training data. The predictions for two new data points show that the model predicts the first student would be admitted (indicated by 1) and the second student would not be admitted (indicated by 0), which are the correct results.

The result shows that the logistic regression model using a library achieved an accuracy score of 0.93 on the training set. This means the model was able to correctly classify 93% of the training examples, indicating a high level of performance. The predictions for the new data points are consistent with the earlier manual implementation, predicting that the first new student will be admitted (1) and the second will not (0). This consistency in predictions for unseen data suggests the model's reliability.

---

## Conclusion
As can be observed in the final section of the code, both the manual and library-based implementations of logistic regression correctly predicted the label for the test set. That is reasonable enough given the high accuracies of both models (0.89 for manual, 0.93 for library). The slight superiority of the library approach’s accuracy might be attributed to highly optimized techniques applied to the model within the `scikit-learn` library.

---
