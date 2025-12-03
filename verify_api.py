import requests
import os
import random
import glob

# Get a random image
root_path = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(root_path, 'raw_data', 'train')
images = glob.glob(os.path.join(train_path, '*.jpg'))

if not images:
    print("No images found for testing.")
    exit(1)

test_image_path = random.choice(images)
print(f"Testing with image: {test_image_path}")

url = "http://127.0.0.1:8000/predict"

with open(test_image_path, 'rb') as f:
    files = {'file': f}
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("✅ API Request Successful")
            print(response.json())
        else:
            print(f"❌ API Request Failed: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")
