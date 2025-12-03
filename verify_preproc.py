from clairvoyance.ml_logic.data import load_data
from clairvoyance.ml_logic.preprocessor import preprocess_images, preprocess_labels
import numpy as np

print("Loading data...")
X, y = load_data('train', n_samples=5)

print(f"Original X shape: {X.shape}")
print(f"Original y: {y}")

print("Preprocessing...")
X_processed = preprocess_images(X)
y_processed = preprocess_labels(y)

print(f"Processed X shape: {X_processed.shape}")
print(f"Processed y: {y_processed}")

if X_processed.shape[1:] == (64, 64, 1):
    print("Shape check passed.")
else:
    print(f"Shape check failed. Expected (N, 64, 64, 1), got {X_processed.shape}")

if X_processed.max() <= 1.0 and X_processed.min() >= 0.0:
    print("Normalization check passed.")
else:
    print("Normalization check failed.")

if y_processed.dtype == np.int32 or y_processed.dtype == np.int64:
    print("Label encoding check passed (integers).")
else:
    print(f"Label encoding check failed. Type: {y_processed.dtype}")
