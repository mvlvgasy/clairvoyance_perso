from clairvoyance.ml_logic.data import load_data
import numpy as np

print("Testing load_data('train')...")
X, y = load_data('train', n_samples=10)

if X is None:
    print("Failed to load data.")
else:
    print(f"Successfully loaded data.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    if len(X) > 0:
        print(f"Sample image shape: {X[0].shape}")
        print(f"Unique labels: {np.unique(y)}")
