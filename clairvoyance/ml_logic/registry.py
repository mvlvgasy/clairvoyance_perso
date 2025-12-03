import os
import time
import pickle
from tensorflow.keras.models import save_model as keras_save_model
from tensorflow.keras.models import load_model as keras_load_model
from clairvoyance.params import *

def save_model(model=None):
    """
    Persist the model to local directory
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(root_path, 'training_outputs', 'models', timestamp)
    print(f"DEBUG: root_path={root_path}")
    print(f"DEBUG: save_path={save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    if model:
        model_path = os.path.join(save_path, "model.h5")
        keras_save_model(model, model_path)
        print(f"✅ Model saved locally at {model_path}")
        
    return None

def load_model(stage="Production"):
    """
    Return a saved model:
    - locally (latest one in training_outputs/models)
    """
    print("\nLoad latest model from local registry...")
    
    local_model_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'training_outputs', 'models')
    
    if not os.path.exists(local_model_directory):
        print("No model directory found")
        return None
        
    # Get latest timestamp
    timestamps = sorted(os.listdir(local_model_directory))
    if not timestamps:
        print("No models found")
        return None
        
    latest_timestamp = timestamps[-1]
    model_path = os.path.join(local_model_directory, latest_timestamp, "model.h5")
    
    if not os.path.exists(model_path):
        print(f"No model file found at {model_path}")
        return None
        
    model = keras_load_model(model_path)
    print("✅ Model loaded from local registry")
    
    return model
