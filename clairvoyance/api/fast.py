from fastapi import FastAPI, UploadFile, File
from clairvoyance.interface.main import pred
import numpy as np
import cv2
from clairvoyance.params import *

app = FastAPI()

@app.get("/")
def index():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return {"error": "Could not decode image"}
        
    # Make prediction (pred expects a list/array of images)
    # We wrap the single image in a list/array
    # Note: pred() handles preprocessing (resize, expand dims, normalize)
    
    # However, pred() expects an array of images.
    # Let's ensure we pass it correctly.
    # Our preprocess_images in preprocessor.py iterates over the input list/array.
    
    prediction = pred([img])
    
    if prediction is None:
         return {"error": "Prediction failed"}
         
    # Return result
    # prediction is (1, num_classes) probabilities
    y_pred = prediction[0]
    class_idx = np.argmax(y_pred)
    class_name = CLASSES[class_idx]
    confidence = float(y_pred[class_idx])
    
    return {
        "class": class_name,
        "confidence": confidence,
        "probabilities": {c: float(p) for c, p in zip(CLASSES, y_pred)}
    }
