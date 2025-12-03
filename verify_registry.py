from clairvoyance.ml_logic.registry import save_model
from tensorflow.keras import Sequential, layers

print("Creating dummy model...")
model = Sequential([layers.Dense(1, input_shape=(10,))])
model.build()

print("Saving model...")
save_model(model)
print("Done.")
