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
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blood Type Prediction</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        /* General Styling */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f4f8;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .main-container {
            background: #ffffff;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            padding: 30px 40px;
            width: 90%;
            max-width: 500px;
            text-align: center;
            overflow: hidden;
        }
        h1 {
            color: #d90429; /* Red theme for blood */
            margin-bottom: 10px;
        }
        p {
            color: #555;
            margin-bottom: 25px;
        }
        /* Upload Container */
        .upload-container {
            margin-bottom: 20px;
        }
        #upload-label {
            border: 3px dashed #cbd5e0;
            border-radius: 10px;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.9s ease;
        }
        #upload-label:hover {
            border-color: #d90429;
            background-color: #fdf2f2;
            transform: scale(1.02);
        }
        #upload-label .fa-fingerprint {
            font-size: 50px;
            color: #d90429;
            margin-bottom: 15px;
            transition: transform 0.3s ease;
        }
        #upload-label:hover .fa-fingerprint {
            transform: scale(1.1);
        }
        #upload-text {
            font-weight: 700;
            color: #4a4a4a;
        }
        /* Image Preview & Scanner Animation */
        #preview-container {
            position: relative;
            border: 2px solid #eee;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        #preview-image {
            max-width: 100%;
            height: auto;
            display: block;
        }
        #scanner-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, transparent, rgba(217, 4, 41, 0.8), transparent);
            box-shadow: 0 0 10px rgba(217, 4, 41, 1);
            display: none;
        }
        @keyframes scan-animation {
            0% { top: 0%; }
            100% { top: 100%; }
        }
        .scanning #scanner-line {
            display: block;
            animation: scan-animation 7s ease-in-out infinite alternate;
        }
        /* Button */
        button {
            font-family: 'Roboto', sans-serif;
            background-color: #d90429;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 25px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            background-color: #b30021;
            box-shadow: 0 5px 15px rgba(217, 4, 41, 0.3);
        }
        button:disabled {
            background-color: #aaa;
            cursor: not-allowed;
        }
        button i {
            margin-right: 8px;
        }
        /* Pop-up Modal */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            animation: fadeIn 0.3s ease;
        }
        .modal-content {
            background: white;
            border-radius: 10px;
            padding: 30px;
            width: 90%;
            max-width: 350px;
            text-align: center;
            position: relative;
            animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55);
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.5); }
            to { opacity: 1; transform: scale(1); }
        }
        .close-button {
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 28px;
            color: #aaa;
            cursor: pointer;
            transition: color 0.2s;
        }
        .close-button:hover {
            color: #333;
        }
        .modal-content h2 {
            color: #333;
            margin-top: 0;
        }
        .modal-content h2 i {
            color: #d90429;
            margin-right: 10px;
        }
        #result-display p {
            font-size: 18px;
            color: #555;
            margin-bottom: 10px;
        }
        #blood-type-result {
            font-size: 72px;
            font-weight: 700;
            color: #d90429;
            line-height: 1;
        }
        /* This is the new element for confidence */
        #confidence-result {
            font-size: 16px; 
            color: #666; 
            margin-top: 10px;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>

    <div class="main-container">
        <h1>Blood Type Prediction</h1>
        <p>Upload a fingerprint image to analyze and predict the blood group.</p>

        <div class="upload-container">
            <input type="file" id="fingerprint-uploader" accept="image/*" hidden>
            <label for="fingerprint-uploader" id="upload-label">
                <i class="fas fa-fingerprint"></i>
                <span id="upload-text">Click or Drag to Upload Fingerprint</span>
            </label>
        </div>

        <div id="preview-container" class="hidden">
            <img id="preview-image" src="#" alt="Fingerprint Preview" />
            <div id="scanner-line"></div>
        </div>

        <button id="predict-button" class="hidden">
            <i class="fas fa-search-plus"></i> Analyze Fingerprint
        </button>
    </div>

    <div id="result-modal" class="modal-overlay hidden">
        <div class="modal-content">
            <span class="close-button">&times;</span>
            <h2><i class="fas fa-vial"></i> Analysis Complete</h2>
            <div id="result-display">
                <p>Predicted Blood Group:</p>
                <div id="blood-type-result">O+</div>
                <p id="confidence-result"></p> 
            </div>
        </div>
    </div>

    <script>
        // Get all the necessary DOM elements
        const uploaderInput = document.getElementById('fingerprint-uploader');
        const uploadLabel = document.getElementById('upload-label');
        const uploadText = document.getElementById('upload-text');
        const previewContainer = document.getElementById('preview-container');
        const previewImage = document.getElementById('preview-image');
        const predictButton = document.getElementById('predict-button');

        const modalOverlay = document.getElementById('result-modal');
        const closeButton = document.querySelector('.close-button');
        const resultText = document.getElementById('blood-type-result');
        // Get the new confidence element
        const confidenceText = document.getElementById('confidence-result'); 

        // 1. Handle the file upload and preview
        uploaderInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('hidden');
                    predictButton.classList.remove('hidden');
                    uploadText.textContent = file.name;
                };
                
                reader.readAsDataURL(file);
            }
        });

        // 2. Handle the "Analyze" button click
        predictButton.addEventListener('click', async () => {
            // Disable the button
            predictButton.disabled = true;
            predictButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

            // Start scanning animation
            previewContainer.classList.add('scanning');

            // Get the file
            const file = uploaderInput.files[0];
            if (!file) {
                resultText.textContent = "Error";
                confidenceText.textContent = "No file selected.";
                modalOverlay.classList.remove('hidden');
                predictButton.disabled = false;
                predictButton.innerHTML = '<i class="fas fa-search-plus"></i> Analyze Fingerprint';
                previewContainer.classList.remove('scanning');
                return;
            }

            // Create FormData to send the file
            const formData = new FormData();
            formData.append('file', file);

            // ---------------------------------
            // !!! THIS IS THE INTEGRATION !!!
            // We replaced the setTimeout simulation with a real fetch call
            // ---------------------------------
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Prediction request failed');
                }

                const data = await response.json();
                
                // --- SUCCESS ---
                // Set the result in the modal
                resultText.textContent = data.predicted_blood_type;
                confidenceText.textContent = Confidence: ${(data.confidence * 100).toFixed(2)}%;

            } catch (error) {
                // --- ERROR ---
                console.error('Error:', error);
                resultText.textContent = 'Error';
                confidenceText.textContent = 'Prediction failed. Please check the console.';
            } finally {
                // --- ALWAYS RUNS ---
                // Stop the scanning animation
                previewContainer.classList.remove('scanning');
                // Show the modal
                modalOverlay.classList.remove('hidden');
                // Re-enable the button
                predictButton.disabled = false;
                predictButton.innerHTML = '<i class="fas fa-search-plus"></i> Analyze Fingerprint';
            }
        });

        // 3. Handle closing the modal
        closeButton.addEventListener('click', () => {
            modalOverlay.classList.add('hidden');
        });

        // 4. Also close the modal if the user clicks on the overlay
        modalOverlay.addEventListener('click', (event) => {
            if (event.target === modalOverlay) {
                modalOverlay.classList.add('hidden');
            }
        });
    </script>
</body>
</html>
    """

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