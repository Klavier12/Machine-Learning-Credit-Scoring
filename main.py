from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Initializing the FastAPI app
app = FastAPI(title="Credit Prediction API")

# Load the model with error handling
try:
    model = joblib.load("model_cat.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Root endpoint to check if API is running
@app.get("/")
def home():
    return {"message": "Credit Prediction API is running!"}

# Define the input data schema
class InputData(BaseModel):
    Age: float
    Credit_Utilization_Rate: float
    Times_30_59_Days_Late: float
    Debt_To_Income_Ratio: float
    Monthly_Income: float
    Open_Credit_Lines_And_Loans: float
    Times_90_Days_Late: float
    Real_Estate_Loans: float
    Times_60_89_Days_Late: float
    Dependents: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input data to a numpy array with shape (1, n_features)
    input_array = np.array([[data.Age,
                             data.Credit_Utilization_Rate,
                             data.Times_30_59_Days_Late,
                             data.Debt_To_Income_Ratio,
                             data.Monthly_Income,
                             data.Open_Credit_Lines_And_Loans,
                             data.Times_90_Days_Late,
                             data.Real_Estate_Loans,
                             data.Times_60_89_Days_Late,
                             data.Dependents]])
    
    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_array)
    
    # Classify: if p(class 0) >= 0.8, assign class 0 (no risk), else class 1 (risk)
    prediction_class = (prediction_proba[:, 0] <= 0.7).astype(int)
    
    return {
        "prediction_proba": prediction_proba.tolist(),
        "prediction": prediction_class.tolist()
    }

# Run the application using uvicorn if executed as main
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8010, reload=True)
