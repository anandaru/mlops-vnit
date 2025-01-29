import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_model():
    """
    Trains a Decision Tree Classifier and saves the model.

    Returns:
        str: A success message if training is completed successfully, or an error message if the dataset is not found.
    """
    # Define paths
    data_path = "./data/Iris.csv"  # Corrected the file path to use forward slashes for cross-platform compatibility
    model_path = "./model/iris_classifier.pkl"  # Corrected the file path to use forward slashes

    # Check if the dataset exists
    if not os.path.exists(data_path):
        return f"Dataset not found at {data_path}"

    # Load the dataset
    df = pd.read_csv(data_path)

    # Features and target variable
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

    # Initialize and train the Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)

    return "Model trained and saved successfully!"
