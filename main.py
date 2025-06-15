from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize app
app = FastAPI(
    title="YouTube Category Prediction API",
    description="API for predicting YouTube video categories based on engagement metrics",
    version="1.0.0"
)



BASE_DIR = "/app/model/YouTubeCategoryModel"


model = joblib.load(os.path.join(BASE_DIR, "best_rf_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# Define input data schema
class VideoFeatures(BaseModel):
    views: float
    likes: float
    dislikes: float
    comment_count: float
    engagement_ratio: float
    like_dislike_ratio: float
    country_encoded: int

    class Config:
        schema_extra = {
            "example": {
                "views": 150000,
                "likes": 7500,
                "dislikes": 150,
                "comment_count": 1200,
                "engagement_ratio": 0.05,
                "like_dislike_ratio": 50.0,
                "country_encoded": 1
            }
        }

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "YouTube Category Prediction API is running!"}

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "OK"}

# Prediction endpoint
@app.post("/predict")
def predict(data: VideoFeatures):
    columns = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "engagement_ratio",
        "like_dislike_ratio",
        "country_encoded"
    ]

    input_df = pd.DataFrame([[
        data.views,
        data.likes,
        data.dislikes,
        data.comment_count,
        data.engagement_ratio,
        data.like_dislike_ratio,
        data.country_encoded
    ]], columns=columns)

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    category_name = label_encoder.inverse_transform([prediction])[0]

    return {
        "predicted_category": int(prediction),
        "category_name": category_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
