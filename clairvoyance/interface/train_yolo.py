from ultralytics import YOLO
import os
from clairvoyance.ml_logic.registry import save_model_to_gcs
from clairvoyance.params import *
import time

def train_yolo():
    """
    Train YOLOv8n model
    """
    print("🚀 Starting YOLOv8n training...")
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (nano version)

    # Path to data.yaml
    # Assuming run from root
    data_yaml_path = os.path.abspath('data.yaml')
    
    # Train the model
    # imgsz=640 is standard for YOLO
    results = model.train(data=data_yaml_path, epochs=10, imgsz=640)
    
    print("✅ Training complete!")
    
    # Upload to GCS if requested
    if MODEL_TARGET == "gcs":
        # results.save_dir gives the path to the run folder (e.g. runs/detect/train2)
        # The best model is usually at {save_dir}/weights/best.pt
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            gcs_path = f"models/{timestamp}/best.pt"
            print(f"☁️ Uploading {best_model_path} to GCS...")
            save_model_to_gcs(best_model_path, gcs_path)
        else:
            print(f"❌ Could not find best model at {best_model_path}")

    return results

if __name__ == '__main__':
    train_yolo()
