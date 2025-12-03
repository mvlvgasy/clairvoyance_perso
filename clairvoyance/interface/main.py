import numpy as np
import pandas as pd
from clairvoyance.ml_logic.data import create_dataset

from clairvoyance.ml_logic.model import initialize_model, compile_model, train_model
from clairvoyance.ml_logic.registry import save_model, load_model
from clairvoyance.params import *

def train():
    """
    Train the model on the local data
    """
    print("\n⭐️ Use case: train")
    
    # 1. Load Data (tf.data.Dataset)
    train_dataset = create_dataset(data_type='train')
    
    # 2. Initialize Model
    # Get one batch to check shape
    for images, labels in train_dataset.take(1):
        input_shape = images.shape[1:]
    
    model = initialize_model(input_shape=input_shape, num_classes=NUM_CLASSES)
    
    # 3. Compile Model
    model = compile_model(model)
    
    # 4. Train Model
    model, history = train_model(model, train_dataset, None, 
                                 epochs=50, 
                                 validation_split=None) # Validation split not supported directly with dataset in this simple setup, would need separate val dataset
    
    # 5. Save
    save_model(model)
    
    return history

def evaluate():
    """
    Evaluate the model on the test data
    """
    print("\n⭐️ Use case: evaluate")
    
    # 1. Load model
    model = load_model()
    if model is None:
        print("No model found.")
        return None
        
    # 2. Load data
    X_test, y_test = load_data(data_type='test')
    if X_test is None:
        print("No test data found.")
        return None
        
    # 3. Preprocess
    X_test_processed = preprocess_images(X_test)
    
    # 4. Evaluate
    metrics = model.evaluate(X_test_processed, y_test)
    print(f"Test metrics: {metrics}")
    
    return metrics

def pred(X_pred):
    """
    Make a prediction using the latest trained model
    """
    print("\n⭐️ Use case: predict")
    
    if X_pred is None:
        print("No data to predict on")
        return None

    model = load_model()
    if model is None:
        print("No model found.")
        return None
        
    # Preprocess
    X_processed = preprocess_images(X_pred)
    
    # Predict
    y_pred = model.predict(X_processed)
    
    print(f"✅ Prediction done: {y_pred}")
    
    return y_pred

if __name__ == '__main__':
    train()
