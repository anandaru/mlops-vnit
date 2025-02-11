from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from src.classifier import train_model
import os

# Initialize FastAPI app
app = FastAPI()

# Corrected Model Path (Matches train_model.py)
MODEL_PATH = "./artifacts/best_iris_classifier.pkl"

# Function to load the model
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    return None

# Load the model at startup
clf = load_model()

# Pydantic model for input validation
class PredictionInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

# Route for checking API status
@app.get("/get_status")
async def get_status():
    return {
        "status": "API is running",
        "training_data_split": "70%",
        "test_data_split": "30%"
    }

# Route for making predictions
@app.post("/prediction")
async def prediction(data: PredictionInput):
    try:
        if clf is None:
            raise HTTPException(status_code=400, detail="Model not loaded. Train the model first using /training endpoint.")

        # Extract input features
        input_data = np.array([[data.SepalLengthCm, data.SepalWidthCm, data.PetalLengthCm, data.PetalWidthCm]])

        # Make prediction
        prediction = clf.predict(input_data)

        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route for retraining the model
@app.get("/training")
async def training():
    try:
        # Call the train_model function to retrain
        message = train_model()

        # Reload the trained model
        global clf
        clf = load_model()

        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
