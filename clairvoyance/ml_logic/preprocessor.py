import numpy as np
import cv2
from clairvoyance.params import *

def preprocess_images(images):
    """
    Preprocess images: Resize, Expand Dims, Normalize
    """
    processed_images = []
    
    for img in images:
        # Resize
        img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Normalize
        img_normalized = img_resized / 255.0
        
        processed_images.append(img_normalized)
        
    return np.array(processed_images)

def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    """
    Encode labels: String -> Integer
    """
    # Create a mapping
    class_to_int = {c: i for i, c in enumerate(CLASSES)}
    
    # Map labels
    encoded_labels = np.array([class_to_int.get(l) for l in labels])
    
    return encoded_labels
