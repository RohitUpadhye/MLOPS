from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Initialize app
app = FastAPI()

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Define request body
class InputData(BaseModel):
    input: list  # Example: [1, 2, 3, 4]

# Home route
@app.get("/")
def home():
    return {"message": "MLOps FastAPI Running 🚀"}

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.input])
    return {"prediction": int(prediction[0])}