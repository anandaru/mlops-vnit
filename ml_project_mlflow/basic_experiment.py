import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
mlflow.set_tracking_uri("http://localhost:5001")
print("MLflow Tracking URI:", mlflow.get_tracking_uri())
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("max_iter", 1000)
    
    # Train model
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    
    # Log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "iris_model")
