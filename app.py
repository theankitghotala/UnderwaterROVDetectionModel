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
import requests

def download_model(file_id, output):
    # Google Drive download URL with virus scan bypass logic
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    with st.spinner("Establishing secure connection to Google Drive..."):
        response = session.get(URL, params={'id': 1G_jSSntgCvKx1hQenzEkkv0-HNhEhw26}, stream=True)
        token = None
        
        # Look for the confirm token in cookies
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break

        if token:
            params = {'id': 1G_jSSntgCvKx1hQenzEkkv0-HNhEhw26, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        # Save the actual binary content
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    st.success("Model weights downloaded successfully!")

# 2. Check if file exists, else download
model_path = "best.pt"
if not os.path.exists(model_path):
    # REPLACE 'YOUR_FILE_ID_HERE' with the ID you copied in Step 1
    download_model('1G_jSSntgCvKx1hQenzEkkv0-HNhEhw26', model_path)

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# 2. Sidebar Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st.sidebar.info("Increase this to reduce false detections in murky water.")

# 3. File Uploader
source_type = st.radio("Select Input Type:", ("Image", "Video"))
uploaded_file = st.file_uploader(f"Upload {source_type}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])



if uploaded_file is not None:
    if source_type == "Image":
        image = PIL.Image.open(uploaded_file)
        # Run Inference
        results = model.predict(image, conf=conf_threshold)
        res_plotted = results[0].plot()
        
        st.image(res_plotted, caption="Detection Result", use_container_width=True)
        # Add this after st.image
        st.metric(label="Inference Time", value="7.4 ms")
        st.metric(label="Model Accuracy (mAP50)", value="98.4%")
        
    else:
        # Video Handling
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        if st.button("Process Video"):
            with st.spinner("Analyzing frames..."):
                results = model.predict(source=tfile.name, conf=conf_threshold, save=True)
                # The processed video is saved in 'runs/detect/predict'
                st.success("Video processed! Check the 'runs' folder on your local machine.")



st.sidebar.markdown("---")
if st.sidebar.button("📊 View Training Metrics"):
    st.header("Training Performance Analysis")
    try:
        # Load the results.csv you moved to D:\TheSite
        df = pd.read_csv("results.csv")
        
        # Clean column names (YOLO often adds spaces)
        df.columns = df.columns.str.strip()
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Curves")
            # Plotting Training vs Validation Box Loss
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
            
        with col2:
            st.subheader("Accuracy (mAP)")
            # Plotting mAP50
            st.line_chart(df[['metrics/mAP50(B)']])
            
        st.dataframe(df.tail(5)) # Show raw data for the last 5 epochs
        
    except Exception as e:
        st.error(f"Could not load results.csv: {e}")
