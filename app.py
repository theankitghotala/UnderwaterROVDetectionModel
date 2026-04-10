import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import PIL.Image
import pandas as pd 
import os
import requests
import io
import time
import numpy as np

# Page Config
st.set_page_config(page_title="ROV Detection Dashboard", layout="wide")
st.title("🚢 Underwater ROV Detection System")
st.sidebar.title("Settings")

# 1. Download Function
def download_model(url, output):
    with st.spinner("Downloading model weights from GitHub..."):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model weights downloaded successfully!")
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")

# 2. Model Loading Logic
# model_url = "https://github.com/theankitghotala/UnderwaterROVDetectionModel/releases/download/v1.0/best.pt" 
# model_path = "best.pt"

model_url = "https://github.com/theankitghotala/UnderwaterROVDetectionModel/releases/download/v2.0/best_new_26.pt" 
model_path = "best_new_26.pt"

if not os.path.exists(model_path):
    download_model(model_url, model_path)

@st.cache_resource
def load_model():
    return YOLO(model_path)

model = load_model()

# 3. Sidebar Controls
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.9)
st.sidebar.info("Increase this to reduce false detections in murky water.")

# 4. File Uploader
source_type = st.radio("Select Input Type:", ("Image", "Video"), horizontal=True)
uploaded_file = st.file_uploader(f"Upload {source_type}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    if source_type == "Image":
        col_input, col_output = st.columns(2)
        
        image = PIL.Image.open(uploaded_file)
        
        with col_input:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)

        with st.status("Analyzing Underwater Environment...", expanded=True) as status:
            st.write("Preprocessing image frames...")
            img_array = np.array(image)
        
            st.write("Running ROV detection model...")
            results = model.predict(image, conf=conf_threshold)

            st.write("Rendering bounding boxes and confidence scores...")
            res_plotted = results[0].plot()
            status.update(label="Detection Complete!", state="complete", expanded=False)
            
        with col_output:
            st.markdown("### Detection Result")
            st.image(res_plotted, use_container_width=True)
            
            # Download Button for the Result
            buf = io.BytesIO()
            PIL.Image.fromarray(res_plotted).save(buf, format="PNG")
            st.download_button(label="📥 Download Result", data=buf.getvalue(), 
                             file_name="rov_detection.png", mime="image/png")
        
        # Detection Summary Stats
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Objects Detected", len(results[0].boxes))
        m2.metric("Inference Time", f"{results[0].speed['inference']:.1f} ms")
        m3.metric("mAP50 Accuracy", "98.4%")
        
    else:
        # Create a temporary file for the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        st.video(tfile.name)
        
        if st.button("Process Video"):
            with st.spinner("Analyzing frames..."):
                # 1. Run Inference
                # 'project' and 'name' ensure we know exactly where the file is saved
                results = model.predict(source=tfile.name, conf=conf_threshold, save=True, project="runs", name="detect", exist_ok=True)
                
                # 2. Locate the processed file
                # YOLO typically saves to runs/detect/video_name.mp4
                processed_path = os.path.join("runs", "detect", os.path.basename(tfile.name))
                
                # 3. Streamlit/Browsers often need H.264 encoding to play MP4s
                # We use OpenCV to read the saved video and re-save it if necessary, 
                # or simply point to the output if it works in your environment.
                if os.path.exists(processed_path):
                    st.success("Video processed successfully!")
                    
                    # Read the processed video file
                    video_file = open(processed_path, 'rb')
                    video_bytes = video_file.read()
                    
                    st.markdown("### Detection Result")
                    st.video(video_bytes)
                    
                    # Download button for the video
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=video_bytes,
                        file_name="detected_rov.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("Processing failed: Result file not found.")
                
                # Cleanup
                os.remove(tfile.name)

# 5. Training Metrics Section 
st.sidebar.markdown("---")
show_metrics = st.sidebar.checkbox("📊 View Training Metrics")

if show_metrics:
    st.header("Technical Training & Performance Analysis")
    
    # Technical Summary
    st.subheader("Objectives & Dataset Summary")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Data Split", "77% Train / 11% Test / 12% Validation")
    col_b.metric("Training Epochs", "100")
    col_c.metric("Optimizer", "MuSGD")
    
    st.markdown("""
    **Core Algorithm:** YOLO (Ultralytics)  
    **Objective:** High-precision object detection for underwater ROV navigation in turbid environments.
    """)

    try:
        # Load and clean results data
        # df = pd.read_csv("results.csv")
        df = pd.read_csv("results_new_26.csv")
        df.columns = df.columns.str.strip()
        
        # Section 1: Loss Curves
        st.subheader("Training vs. Testing (Validation) Loss")
        st.info("These curves show the reduction in error (loss) across 100 epochs.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Box Loss (Localization)**")
            st.line_chart(df[['train/box_loss', 'val/box_loss']])
        with c2:
            st.write("**Class Loss (Identification)**")
            st.line_chart(df[['train/cls_loss', 'val/cls_loss']])
            
        # Section 2: Accuracy Metrics
        st.subheader("Mean Average Precision (mAP)")
        st.line_chart(df[['metrics/mAP50(B)', 'metrics/mAP50-95(B)']])
        
        st.write("**Full Epoch Logs (Last 10)**")
        st.dataframe(df.tail(10)) 
        
    except Exception as e:
        st.error(f"Could not load results.csv: {e}. Ensure the file is in your GitHub root.")
