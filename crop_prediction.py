import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import chardet

with open(r'C:\Users\ronit\OneDrive\Desktop\College\Python\Hacker\generated_data.csv', 'rb') as f:
    result = chardet.detect(f.read())

data = pd.read_csv(r'C:\Users\ronit\OneDrive\Desktop\College\Python\Hacker\generated_data.csv', encoding=result['encoding'])

X = data.drop(columns=['Crop Recommended'])  # Features
y = data['Crop Recommended']  # Target variable

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Train the classifier on the training data
mlp_classifier.fit(X_train, y_train)

# Reorder the columns of the test dataset to match the order of columns in the training dataset
X_test = X_test[X_train.columns]

# Make predictions on the testing data
y_pred = mlp_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the MLP classifier: {accuracy:.2f}")

# Predict the recommended crop for a specific farm ID in Haryana
farm_id_to_predict = 1  # Replace with the desired farm ID
farm_data = data[data['Farm ID'] == farm_id_to_predict].drop(columns=['Crop Recommended', 'Farm ID'])
farm_data = pd.get_dummies(farm_data)

# Add missing columns to farm_data
missing_columns = set(X_train.columns) - set(farm_data.columns)
for col in missing_columns:
    farm_data[col] = 0

# Reorder the columns of the farm data to match the order of columns in the training dataset
farm_data = farm_data[X_train.columns]

recommended_crop = mlp_classifier.predict(farm_data)
print(f"Recommended crop for Farm ID {farm_id_to_predict}: {recommended_crop[0]}")