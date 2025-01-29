from flask import Flask, request, jsonify
import pickle
import numpy as np
from classifier import train_model
import os

# Initialize Flask app
app = Flask(__name__)

# Model path
MODEL_PATH = "./model/iris_classifier.pkl"

# Load the trained model if it exists
clf = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        clf = pickle.load(file)

# Route for checking API status
@app.route("/get_status", methods=["GET"])
def get_status():
    return jsonify({"status": "API is running", "training_data_split": "70%", "test_data_split": "30%"})

# Route for making predictions
@app.route("/prediction", methods=["POST"])
def prediction():
    try:
        if clf is None:
            return jsonify({"error": "Model not loaded. Train the model first using /training endpoint."}), 400

        # Get JSON payload
        payload = request.json

        # Extract input features
        input_data = np.array([[payload["SepalLengthCm"], payload["SepalWidthCm"], payload["PetalLengthCm"], payload["PetalWidthCm"]]])

        # Make prediction
        prediction = clf.predict(input_data)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for retraining the model
@app.route("/training", methods=["GET"])
def training():
    try:
        # Call the train_model function
        global clf
        message = train_model()

        # Reload the trained model
        with open(MODEL_PATH, 'rb') as file:
            clf = pickle.load(file)

        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
