from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from clairvoyance.params import *

def initialize_model(input_shape, num_classes=3):
    """
    Initialize the Neural Network with Data Augmentation and more complexity
    """
    model = Sequential()
    
    # Data Augmentation Layer
    model.add(layers.Input(shape=input_shape))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    # First Convolution Block
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Second Convolution Block
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Third Convolution Block (New)
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(layers.Flatten())
    
    # Dense Layers with Dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Reduce overfitting
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    print("✅ Model initialized with Data Augmentation and Deeper Architecture")
    
    return model

def compile_model(model, learning_rate=0.001):
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_model(model, X, y, batch_size=BATCH_SIZE, epochs=50, validation_split=0.2):
    """
    Train the Neural Network with Callbacks
    """
    es = EarlyStopping(patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
    
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[es, rlr],
                        verbose=1)
    return model, history
