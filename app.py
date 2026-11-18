import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference

from Train import train

# the default Swagger page is turned off, Scalar is shown at "/" instead
app = FastAPI(title="Student Pass/Fail API", docs_url=None, redoc_url=None)


class Student(BaseModel):
    study_hours: float
    attendance: float
    previous_marks: float


@app.get("/", include_in_schema=False)
def scalar_ui():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title="Student Pass/Fail API")


@app.post("/predict", summary="Predict Pass or Fail")
def predict(student: Student):
    model = joblib.load("Models/model.pkl")
    features = [[student.study_hours, student.attendance, student.previous_marks]]
    result = model.predict(features)[0]
    return {"result": result}


@app.post("/train", summary="Train the models")
def train_models():
    # trains the models, logs them to MLflow and saves the best one to models/model.pkl
    return {"runs": train()}