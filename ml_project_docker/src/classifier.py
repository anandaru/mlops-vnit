import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import pickle
import os
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")  # Make sure MLflow is running on this port
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def train_model():
    """
    Trains a Decision Tree Classifier using hyperparameter tuning, logs all experiments in MLflow,
    selects the best model based on accuracy, and saves it.

    Returns:
        str: A success message with accuracy and best hyperparameters.
    """
    # Define correct paths
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Base directory of src/
    data_path = os.path.join(base_dir, "../data/Iris.csv")  # Ensure dataset path is correct
    model_path = os.path.join(base_dir, "../artifacts/best_iris_classifier.pkl")  # Save model properly

    # Ensure MLflow experiment is set
    mlflow.set_experiment("iris_classification_anand1")

    # Check if dataset exists
    if not os.path.exists(data_path):
        return f"❌ Dataset not found at {data_path}"

    # Load dataset
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return f"❌ Error loading dataset: {str(e)}"

    # Check required columns
    required_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    if not all(col in df.columns for col in required_columns):
        return "❌ Dataset is missing required columns!"

    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Species']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True, random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, None],
        'criterion': ['gini', 'entropy']
    }
    
    best_model = None
    best_accuracy = 0
    best_params = None

    # Iterate over all hyperparameter combinations
    for params in ParameterGrid(param_grid):
        with mlflow.start_run(nested=True):  # Nested runs for better tracking
            try:
                # Train model with specific hyperparameters
                clf = DecisionTreeClassifier(**params, random_state=42)
                clf.fit(X_train, y_train)

                # Make predictions
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Log parameters and accuracy
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.sklearn.log_model(clf, "decision_tree_model")

                # Store the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = clf
                    best_params = params

            except Exception as e:
                print(f"❌ Error in training with params {params}: {str(e)}")

    if best_model is None:
        return "❌ No model was successfully trained."

    # Save the best trained model locally
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)

    # Log the best model separately
    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "best_decision_tree_model")
        mlflow.log_params(best_params)
        mlflow.log_metric("best_accuracy", best_accuracy)

    return f"✅ Best model trained, saved, and logged with MLflow! Accuracy: {best_accuracy:.4f}, Best Params: {best_params}"

# Run the function
if __name__ == "__main__":
    print(train_model())
