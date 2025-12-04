import os
import time
import pickle
import glob
from tensorflow.keras.models import save_model as keras_save_model
from tensorflow.keras.models import load_model as keras_load_model
from google.cloud import storage
from clairvoyance.params import *

def save_model(model=None, params=None, metrics=None):
    """
    Persist the model locally and optionally to GCS
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(root_path, 'training_outputs', 'models', timestamp)
    os.makedirs(save_path, exist_ok=True)
    
    # Save params locally
    if params is not None:
        params_path = os.path.join(save_path, "params.pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(save_path, "metrics.pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
    
    if model:
        # Check if it's a YOLO model (Ultralytics) or Keras
        if hasattr(model, 'export'): # YOLO model
             # YOLO models are usually saved as .pt files during training
             # If we are passed the path to the best.pt, we can use that
             pass 
        else:
            model_path = os.path.join(save_path, "model.h5")
            keras_save_model(model, model_path)
            print(f"✅ Model saved locally at {model_path}")
            
            if MODEL_TARGET == "gcs":
                save_model_to_gcs(model_path, f"models/{timestamp}/model.h5")

    return None

def save_model_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Upload a file to GCS
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ File saved to GCS: gs://{BUCKET_NAME}/{gcs_path}")

def load_model(stage="Production"):
    """
    Return a saved model:
    - locally (latest one in training_outputs/models)
    - or from GCS (latest one)
    """
    if MODEL_TARGET == "local":
        print("\nLoad latest model from local registry...")
        
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_model_directory = os.path.join(root_path, 'training_outputs', 'models')
        
        if not os.path.exists(local_model_directory):
            print("No model directory found")
            return None
            
        timestamps = sorted(os.listdir(local_model_directory))
        if not timestamps:
            print("No models found")
            return None
            
        latest_timestamp = timestamps[-1]
        model_path = os.path.join(local_model_directory, latest_timestamp, "model.h5")
        
        # Check for YOLO .pt file if .h5 doesn't exist
        if not os.path.exists(model_path):
             yolo_path = os.path.join(local_model_directory, latest_timestamp, "best.pt")
             if os.path.exists(yolo_path):
                 from ultralytics import YOLO
                 print(f"✅ YOLO Model loaded from {yolo_path}")
                 return YOLO(yolo_path)
        
        if not os.path.exists(model_path):
            print(f"No model file found at {model_path}")
            return None
            
        model = keras_load_model(model_path)
        print("✅ Model loaded from local registry")
        return model

    elif MODEL_TARGET == "gcs":
        print(f"\nLoad latest model from GCS ({BUCKET_NAME})...")
        
        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))
        
        if not blobs:
             print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
             return None

        latest_blob = sorted(blobs, key=lambda x: x.updated)[-1]
        
        # Download to local
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_model_path = os.path.join(root_path, 'training_outputs', 'models', 'gcs_downloaded')
        os.makedirs(local_model_path, exist_ok=True)
        
        filename = os.path.basename(latest_blob.name)
        save_path = os.path.join(local_model_path, filename)
        
        latest_blob.download_to_filename(save_path)
        print(f"✅ Model downloaded from GCS to {save_path}")
        
        if save_path.endswith(".pt"):
             from ultralytics import YOLO
             return YOLO(save_path)
        else:
             return keras_load_model(save_path)
             
    return None
