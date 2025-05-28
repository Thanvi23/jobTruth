from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
# import torch
from utils import prediction_model
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class text(BaseModel):
    title: str
    location: str
    department: str
    salary_range: str
    company_profile: str
    description: str
    requirements: str
    benefits: str

model = prediction_model()

@app.post("/predict")
async def predict_blood_group(text: text):
    text = text.title + " " + text.location + " " + text.department + " " + text.salary_range + " " + text.company_profile + " " + text.description + " " + text.requirements + " " + text.benefits
    prediction = model.predict(text)
    print(prediction)
    return JSONResponse(content={"prediction": prediction*100})
    # return prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)