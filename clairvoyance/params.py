import os
import numpy as np

##################  VARIABLES  ##################
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 128))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
DATA_SOURCE = os.environ.get("DATA_SOURCE", "local") # 'local' or 'gcs'
MODEL_TARGET = os.environ.get("MODEL_TARGET", "local") # 'local' or 'gcs'
BUCKET_NAME = os.environ.get("BUCKET_NAME", "clairvoyance-models-yolo")

CLASSES = ['bus', 'car', 'motorcycle', 'truck']
NUM_CLASSES = len(CLASSES)
