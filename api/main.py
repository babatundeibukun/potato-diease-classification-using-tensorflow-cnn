from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.keras.models.load_model("../saved_model/potatoes.h5")
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
    predictions = MODEL.predict(img_batch, verbose=0)  # Get prediction
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {"prediction": predicted_class, "confidence" : float(confidence)}  # Convert prediction to list for JSON response
    pass
