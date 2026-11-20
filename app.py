from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mltracker import Train


MODEL_PATH = Path(__file__).parent / "Models" / "model.pkl"
app = FastAPI(title="Student Pass/Fail API", docs_url="/docs", redoc_url=None)
@app.get("/")
def home():
    return {
        "message": "I HATE ASMITHA",
        "docs": "/docs",
        "endpoints": ["/predict", "/train"]
    }

class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_marks: float = Field(ge=0, le=100)


@app.post("/predict", summary="Predict Pass or Fail")
def predict(student: Student):
    try:
        model = joblib.load(MODEL_PATH)
        features = [[student.study_hours, student.attendance, student.previous_marks]]
        prediction = str(model.predict(features)[0]).strip().lower()

        if prediction in {"pass", "1", "true"}:
            result = "Pass"
        elif prediction in {"fail", "0", "false"}:
            result = "Fail"
        else:
            raise HTTPException(
                status_code=500, detail=f"Unknown model prediction: {prediction}"
            )

        return {"prediction": result, "student_data": student.model_dump()}
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Model not found. Train the model first using POST /train.",
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")


@app.post("/train", summary="Train the models")
def train_models():
    try:
        return {"message": "Training completed successfully", "runs": Train()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file was not found.")
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Training failed: {error}")
