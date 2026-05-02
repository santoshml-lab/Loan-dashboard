from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# 🔥 CORS import
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("loan_model.joblib")

app = FastAPI()

# 🔥 CORS setup (important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production में specific domain देना
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class InputData(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.get("/")
def home():
    return {"message": "Loan Prediction API Running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    import pandas as pd

    input_df = pd.DataFrame([{
        "no_of_dependents": data.no_of_dependents,
        "education": data.education,
        "self_employed": data.self_employed,
        "income_annum": data.income_annum,
        "loan_amount": data.loan_amount,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value
    }])

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction),
        "result": "Approved ✅" if prediction == 1 else "Rejected ❌"
    }

