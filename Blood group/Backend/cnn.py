from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import shutil
import os
from scalar_fastapi import get_scalar_api_reference

# ---------------------------
# Load the trained CNN model
# ---------------------------
# Load from the same directory (more portable than a D: drive path)
MODEL_PATH = r"D:\Projects\Minor Project\coding\cnn_model_sairam.h5" 
model = tf.keras.models.load_model(MODEL_PATH)

# Blood group labels (same as training)
class_labels = ['A+', 'A-', 'AB-', 'AB+', 'B+', 'B-', 'O+', 'O-']

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="Blood Group Prediction API")

# ---------------------------
# HTML Upload Page (Your Advanced Frontend)
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def upload_form():
    # We've pasted your entire "frontend model" HTML here
    return ""
# ---------------------------
# Prediction Endpoint (Unchanged)
# ---------------------------
@app.post("/predict")
async def predict_blood_group(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and preprocess image
        img = image.load_img(temp_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        predicted_index = np.argmax(pred, axis=1)[0]
        predicted_class = class_labels[predicted_index]
        confidence = float(pred[0][predicted_index])

        # Return JSON result
        return JSONResponse({
            "predicted_blood_type": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"message": "Error processing image"})
    finally:
        # Delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------------------
# Scalar API Reference (Unchanged)
# ---------------------------
@app.get("/scalar")
async def scalar_docs():
    return get_scalar_api_reference(openapi_url=app.openapi_url, title=app.title)

# ---------------------------
# Run Server
# ---------------------------
if __name__== "_main_":
    uvicorn.run(app, host="127.0.0.1", port=8000)