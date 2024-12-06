from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8600/v1/models/potato_model/versions/3:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello , I deploying my first deep learning model , the first of many to come"


def read_file_as_image(data):
    # Ensure data is treated as binary
    image = Image.open(BytesIO(data))  # Open the image using BytesIO
    return np.array(image)  # Convert the image to a numpy array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()  # Read the binary data
    image = read_file_as_image(data)
    img_batch = np.expand_dims(image, 0)  # Add batch dimension

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)

    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

    pass
