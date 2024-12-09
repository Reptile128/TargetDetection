# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
# Replace 'your_dataset.csv' with the actual dataset file
data = pd.read_csv("Data/Tweets_edited_wo_translation_correction_stopwordremoval.csv")

# Step 2: Split dataset into features (X) and target labels (y)
X = data.iloc[1].values  # Features (all columns except the last)
y = data.iloc[2].values   # Target label (last column)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Support Vector Machine Classifier
classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)  # Initialize the model
classifier.fit(X_train, y_train)  # Train the model

# Step 5: Make predictions
y_pred = classifier.predict(X_test)  # Predict on test data

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy: {accuracy:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # Confusion Matrix

print("Classification Report:")
print(classification_report(y_test, y_pred))  # Detailed performance metrics
