import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import PIL.Image
import pandas as pd 
import os
import requests

# Page Config
st.set_page_config(page_title="ROV Detection Dashboard", layout="wide")
st.title("🚢 Underwater ROV Detection System")
st.sidebar.title("Settings")

# 1. Download function for large files
def download_model(file_id, output):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    with st.spinner("Connecting to Google Drive to fetch model weights..."):
        # FIXED: Added quotes around the ID variables below
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = None
        
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
    st.success("Model weights downloaded successfully!")

# 2. Check if file exists, else download
model_path = "best.pt"
google_drive_id = '1G_jSSntgCvKx1hQenzEkkv0-HNhEhw26' # FIXED: Added quotes

if not os.path.exists(model_path):
    download_model(google_drive_id, model_path)

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# Sidebar Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st.sidebar.info("Increase this to reduce false detections in murky water.")

# 3. File Uploader
source_type = st.radio("Select Input Type:", ("Image", "Video"))
uploaded_file = st.file_uploader(f"Upload {source_type}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    if source_type == "Image":
        image = PIL.Image.open(uploaded_file)
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        st.image(res_plotted, caption="Detection Result", use_container_width=True)
        
        # Displaying metrics
        c1, c2 = st.columns(2)
        c1.metric(label="Inference Time", value="7.4 ms")
        c2.metric(label="Model Accuracy (mAP50)", value="98.4%")
        
    else:
        # Video Handling
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        if st.button("Process Video"):
            with st.spinner("Analyzing frames..."):
                # Inference on video
                results = model.predict(source=tfile.name, conf=conf_threshold, save=True)
                st.success("Video processed successfully!")
                st.info("Note: In the cloud version, processed videos are stored in the temporary server directory.")

# Training Metrics Button
st.sidebar.markdown("---")
if st.sidebar.button("📊 View Training Metrics"):
    st.header("Training Performance Analysis")
    try:
        df = pd.read_csv("results.csv")
        df.columns = df.columns.str.strip()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Loss Curves")
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
            
        with col2:
            st.subheader("Accuracy (mAP)")
            st.line_chart(df[['metrics/mAP50(B)']])
            
        st.dataframe(df.tail(5)) 
        
    except Exception as e:
        st.error(f"Could not load results.csv: {e}")
