import os
import pandas as pd
import numpy as np
import cv2
from google.cloud import storage
from clairvoyance.params import *

import tensorflow as tf

def parse_image(filename, label):
    """
    TensorFlow function to load and preprocess images
    """
    img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0  # Normalize
    return img, label

def create_dataset(data_type='train', batch_size=BATCH_SIZE):
    """
    Create a tf.data.Dataset from local files
    """
    # Path to the data
    if DATA_SOURCE == 'local':
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(root_path, 'raw_data', data_type)
    else:
        pass

    # Load CSV
    csv_path = os.path.join(data_path, '_annotations.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter for single object images
    image_counts = df['filename'].value_counts()
    single_object_images = image_counts[image_counts == 1].index
    df_filtered = df[df['filename'].isin(single_object_images)]
    
    # Create full paths
    file_paths = df_filtered['filename'].apply(lambda x: os.path.join(data_path, x)).values
    
    # Encode labels
    class_to_int = {c: i for i, c in enumerate(CLASSES)}
    labels = df_filtered['class'].map(class_to_int).values
    
    print(f"Creating dataset with {len(file_paths)} images from {data_path}...")
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Parallel processing
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and batch
    if data_type == 'train':
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset
