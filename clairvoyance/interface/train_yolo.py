from ultralytics import YOLO
import os

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
    return results

if __name__ == '__main__':
    train_yolo()
