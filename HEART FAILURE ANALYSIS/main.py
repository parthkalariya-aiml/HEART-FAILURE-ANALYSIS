from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI(title="Heart Failure Readmission API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('readmission_pipeline.pkl')

class PatientData(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Smoking_Status: str
    Alcohol_Consumption: str
    Exercise_Frequency: int
    Hypertension: int
    Diabetes: int
    Chronic_Kidney_Disease: int
    Coronary_Artery_Disease: int
    Previous_Stroke: int
    Atrial_Fibrillation: int
    Previous_HF_Admissions: int
    Previous_Hospital_Admissions: int
    Heart_Failure_Type: str
    NYHA_Class: int
    Ejection_Fraction: float
    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float
    Oxygen_Saturation: float
    Creatinine: float
    Sodium: float
    Potassium: float
    Hemoglobin: float
    Blood_Glucose: float
    Emergency_Admission: int
    Beta_Blocker: int
    Length_of_Stay: int
    ICU_Admission: int
    BNP: float
    ACE_ARB: int
    Diuretic: int
    SGLT2_Inhibitor: int

@app.post("/predict")
def predict_single(patient: PatientData):
    try:
        # 1. Convert the validated web data to a simple dictionary
        p_dict = patient.dict()
        
        # 2. Extract the exact column strings the model was trained on
        expected_cols = model.feature_names_in_
        
        # 3. Dynamically align the data (Translates spaces/underscores automatically)
        aligned_data = {}
        for col in expected_cols:
            if col in p_dict:
                aligned_data[col] = p_dict[col]
            elif col.replace(' ', '_') in p_dict:
                aligned_data[col] = p_dict[col.replace(' ', '_')]
            elif col.replace('_', ' ') in p_dict:
                aligned_data[col] = p_dict[col.replace('_', ' ')]
            else:
                return {"API_STATUS": "FAILED", "REASON": f"Missing column: {col}"}
                
        # 4. Create the final DataFrame mapped perfectly for Scikit-Learn
        data_df = pd.DataFrame([aligned_data])
        
        # 5. Execute Prediction
        proba = model.predict_proba(data_df)[0][1]
        
        return {
            "readmission_probability": float(proba),
            "risk_level": "High Risk" if proba > 0.5 else "Low Risk"
        }
        
    except Exception as e:
        # 6. If ANYTHING fails, block the 500 error and send the log to the frontend
        return {
            "API_STATUS": "FAILED",
            "REASON": str(e),
            "TRACE": traceback.format_exc()
        }