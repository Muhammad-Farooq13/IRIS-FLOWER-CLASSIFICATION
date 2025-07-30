
"""# Task
Train a machine learning model to classify iris species based on their measurements using the dataset from "/content/Iris.csv".

## Load the dataset

### Subtask:
Load the Iris dataset into a pandas DataFrame.

**Reasoning**:
Import pandas, load the CSV file into a DataFrame, and display the head of the DataFrame to verify.
"""

import pandas as pd

df = pd.read_csv('/content/Iris.csv')
display(df.head())

"""## Prepare the data

### Subtask:
Split the data into features (measurements) and target (species).

**Reasoning**:
Split the data into features (X) and target (y) as instructed.
"""

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']
display(X.head())
display(y.head())

"""## Split data for training and testing

### Subtask:
Divide the dataset into training and testing sets.

**Reasoning**:
Split the data into training and testing sets and print their shapes.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

"""## Choose and train a model

### Subtask:
Select a suitable classification model (e.g., Support Vector Machine) and train it using the training data.

**Reasoning**:
Import the necessary class, instantiate the model, and train it using the training data.
"""

from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)

"""## Evaluate the model

### Subtask:
Assess the performance of the trained model using the testing data and appropriate metrics.

**Reasoning**:
Use the trained model to make predictions on the test data and calculate the accuracy score.
"""

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy:.2f}")

"""## Summary:

### Data Analysis Key Findings

*   The dataset was successfully loaded into a pandas DataFrame.
*   The data was split into features (measurements) and the target (species).
*   The dataset was divided into training (120 samples) and testing (30 samples) sets, with a test size of 20%.
*   A Support Vector Machine (SVC) model with a linear kernel was chosen and trained on the training data.
*   The trained model achieved an accuracy of 1.00 on the testing data.

### Insights or Next Steps

*   The trained SVC model achieved perfect accuracy on the test set, suggesting it is highly effective for this dataset.
*   Further analysis with different models or cross-validation could be performed to ensure the model's robustness and generalize ability beyond this specific test set split.

"""
