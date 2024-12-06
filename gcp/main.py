from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import jsonify



model = None
interpreter = None
input_index = None
output_index = None

class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "revolvingcity_tf_model"  # Here you need to put the name of your GCP bucket


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


def predict(request):
    global model
    try:
        # Load the model if not already loaded
        if model is None:
            download_blob(
                BUCKET_NAME,
                "models/potatoes.h5",
                "/tmp/potatoes.h5",
            )
            model = tf.keras.models.load_model("/tmp/potatoes.h5")

        # Ensure the file is in the request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400  # Bad Request

        # Process the uploaded image
        image = request.files["file"]
        image = np.array(
            Image.open(image).convert("RGB").resize((256, 256))  # Image resizing
        )
        #image = image / 255  # Normalize the image
        #the line of code above  made the class returned to be the same irrespective of the the image

        # Make prediction
        img_array = tf.expand_dims(image, 0)
        predictions = model.predict(img_array)

        print("Predictions:", predictions)  # Log predictions for debugging

        # Interpret the results
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(round(100 * (np.max(predictions[0])), 2))  # Convert to Python float

        # Return the prediction results
        return jsonify({"class": predicted_class, "confidence": confidence}), 200  # OK

    except Exception as e:
        # Log the error for debugging
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500  # Internal Server Error