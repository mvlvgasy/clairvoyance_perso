import streamlit as st
import requests
from PIL import Image
import numpy as np

# Set API URL
# If running locally with Docker or Uvicorn
API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Clairvoyance - Vehicle Detection",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Clairvoyance - Vehicle Detection")
st.markdown("Upload an image to detect if there is a car.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width='stretch')
    
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            # Prepare file for API
            # We need to send the raw bytes
            # Reset file pointer
            uploaded_file.seek(0)
            files = {'file': uploaded_file}
            
            try:
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        class_name = result['class']
                        confidence = result['confidence']
                        probabilities = result['probabilities']
                        
                        st.success(f"Prediction: **{class_name.upper()}** ({confidence:.2%})")
                        
                        st.bar_chart(probabilities)
                        
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
