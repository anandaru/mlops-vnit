#### **Hyperparameter Tuning with Autologging**

import mlflow
import mlflow.sklearn  # Ensure the sklearn module is imported for logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {"n_estimators": [10, 50, 100], "max_depth": [3, 5, None]}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring="accuracy")

# Start MLflow run
with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Log best model accuracy
    mlflow.log_metric("best_accuracy", grid_search.best_score_)

    # Log the best model explicitly
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_}")
