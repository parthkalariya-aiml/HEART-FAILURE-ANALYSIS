# Heart Failure Readmission Risk Predictor 🫀

An end-to-end machine learning system designed to predict 30-day hospital readmission risk for heart failure patients. This project demonstrates the full lifecycle of an AI application: from data preprocessing and model training to deploying a production-ready REST API and an interactive web frontend.

## 🚀 Project Overview
Hospital readmissions are a critical metric for healthcare quality and cost. This system analyzes 34 distinct clinical features (e.g., Blood Glucose, Ejection Fraction, medications) to output a probability score and risk classification (High/Low Risk) for incoming patients.

### Tech Stack
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib (Logistic Regression pipeline)
* **Backend API:** FastAPI, Pydantic, Uvicorn 
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
* **Data:** 12,000+ patient medical records

## 🧠 System Architecture
1. **Model Pipeline:** The ML pipeline handles missing value imputation, feature scaling, and binary classification. The finalized model is serialized as `readmission_pipeline.pkl`.
2. **FastAPI Backend:** A strict data validation layer (via Pydantic) ensures incoming JSON payloads perfectly match the `scikit-learn` expected schema. Includes automated dynamic column mapping and global exception handling.
3. **Client Interface:** A responsive, dynamically generated web form that submits patient diagnostics to the API and renders the resulting AI assessment.

## 💻 Local Setup & Execution

**1. Clone the repository**
\`\`\`bash
https://github.com/parthkalariya-aiml/HEART-FAILURE-ANALYSIS
cd heart-failure-predictor
\`\`\`

**2. Install dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**3. Run the backend server**
\`\`\`bash
python -m uvicorn main:app --reload
\`\`\`
*The API will be available at `http://127.0.0.1:8000`. You can test endpoints interactively at `http://127.0.0.1:8000/docs`.*

**4. Launch the frontend**
Open `index.html` directly in any web browser to access the prediction interface.

---
*Created by: Kalariya Parth | B.Tech CE-AIML*
