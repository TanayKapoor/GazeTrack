import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Add gaze_app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gaze_app'))

from config import data_config
from utils.helpers import get_model

# Streamlit configuration
st.set_page_config(page_title="Gaze Estimation Demo", layout="wide")

# Simple face detection using OpenCV (no external dependencies)
def detect_faces_opencv(image):
    """Detect faces using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    bboxes = []
    for (x, y, w, h) in faces:
        # Convert to [x_min, y_min, x_max, y_max, confidence] format
        bboxes.append([x, y, x+w, y+h, 0.9])
    
    return bboxes

@st.cache_resource
def load_gaze_model():
    """Load gaze estimation model"""
    try:
        # Model configuration for gaze360 dataset
        dataset_config = data_config["gaze360"]
        bins = dataset_config["bins"]
        binwidth = dataset_config["binwidth"] 
        angle = dataset_config["angle"]
        
        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
        
        # Load gaze estimation model
        weight_path = os.path.join("gaze_app", "weights", "mobileone_s0.pt")
        if not os.path.exists(weight_path):
            st.error(f"❌ Model weights not found at {weight_path}")
            return None, None, None, None, None
            
        gaze_detector = get_model("mobileone_s0", bins, inference_mode=True)
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        gaze_detector.load_state_dict(state_dict)
        gaze_detector.to(device)
        gaze_detector.eval()
        
        st.success("✅ Gaze estimation model loaded successfully!")
        return gaze_detector, device, idx_tensor, binwidth, angle
        
    except Exception as e:
        st.error(f"❌ Error loading gaze model: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None, None, None, None, None

def preprocess_face(face_image):
    """Preprocess face image for gaze estimation"""
    if len(face_image.shape) == 3 and face_image.shape[2] == 3:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_tensor = transform(face_image)
    return face_tensor.unsqueeze(0)

def draw_gaze_on_image(image, bbox, pitch, yaw, thickness=3, color=(0, 255, 0)):
    """Draw gaze direction on image"""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    
    # Calculate center of face
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # Calculate gaze direction
    length = max(x_max - x_min, 60)  # Minimum arrow length
    dx = int(-length * 0.8 * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * 0.8 * np.sin(yaw))
    
    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)
    
    # Draw face bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw gaze direction
    cv2.circle(image, point1, radius=5, color=color, thickness=-1)
    cv2.arrowedLine(image, point1, point2, color=color, thickness=thickness, 
                   line_type=cv2.LINE_AA, tipLength=0.3)
    
    # Add face ID text
    cv2.putText(image, f"Face", (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, color, 2)
    
    return image

def process_image(uploaded_image):
    """Process uploaded image for gaze estimation"""
    
    # Load gaze model
    model_data = load_gaze_model()
    if model_data[0] is None:
        return None, "Failed to load gaze estimation model"
    
    gaze_detector, device, idx_tensor, binwidth, angle = model_data
    
    try:
        # Convert PIL to OpenCV format
        image = np.array(uploaded_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces using OpenCV
        bboxes = detect_faces_opencv(image)
        
        if len(bboxes) == 0:
            return None, "No faces detected in the image"
        
        results = []
        processed_image = image.copy()
        
        # Process each detected face
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            confidence = bbox[4] if len(bbox) > 4 else 1.0
            
            # Extract face region with some padding
            padding = 10
            y_min_pad = max(0, y_min - padding)
            y_max_pad = min(image.shape[0], y_max + padding)
            x_min_pad = max(0, x_min - padding)
            x_max_pad = min(image.shape[1], x_max + padding)
            
            face_region = image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
            
            if face_region.size == 0 or face_region.shape[0] < 30 or face_region.shape[1] < 30:
                continue
                
            # Preprocess face
            face_tensor = preprocess_face(face_region)
            face_tensor = face_tensor.to(device)
            
            # Predict gaze
            with torch.no_grad():
                pitch_pred, yaw_pred = gaze_detector(face_tensor)
                
                # Apply softmax and convert to angles
                pitch_prob = F.softmax(pitch_pred, dim=1)
                yaw_prob = F.softmax(yaw_pred, dim=1)
                
                pitch_angle = torch.sum(pitch_prob * idx_tensor, dim=1) * binwidth - angle
                yaw_angle = torch.sum(yaw_prob * idx_tensor, dim=1) * binwidth - angle
                
                # Convert to radians for visualization
                pitch_rad = np.radians(pitch_angle.cpu().numpy()[0])
                yaw_rad = np.radians(yaw_angle.cpu().numpy()[0])
                
                # Draw gaze on image
                processed_image = draw_gaze_on_image(processed_image, bbox, pitch_rad, yaw_rad)
                
                results.append({
                    'face_id': i + 1,
                    'confidence': confidence,
                    'pitch_degrees': float(pitch_angle.cpu().numpy()[0]),
                    'yaw_degrees': float(yaw_angle.cpu().numpy()[0]),
                    'bbox': bbox
                })
        
        # Convert back to RGB for display
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image, results
        
    except Exception as e:
        import traceback
        return None, f"Error processing image: {str(e)}\n{traceback.format_exc()}"

# Main UI
st.title("🎯 Gaze Estimation Demo")
st.markdown("**Upload an image to detect faces and estimate gaze direction using MobileOne**")

# Check if gaze_app directory exists
if not os.path.exists("gaze_app"):
    st.error("❌ gaze_app directory not found!")
    st.error("Make sure you're running from the correct directory")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image with faces for gaze estimation"
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Image")
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Add some info about the image
        st.info(f"📊 Image size: {image.size[0]} x {image.size[1]} pixels")
        
        if st.button("🎯 Estimate Gaze", type="primary", use_container_width=True):
            with st.spinner("🔍 Detecting faces and estimating gaze directions..."):
                processed_image, results = process_image(image)
                
                if processed_image is not None:
                    st.session_state.processed_image = processed_image
                    st.session_state.results = results
                    st.success(f"✅ Found {len(results)} face(s) with gaze estimation!")
                else:
                    st.error(f"❌ {results}")
    
    with col2:
        st.subheader("📊 Results")
        
        if hasattr(st.session_state, 'processed_image') and st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, caption="Gaze Estimation Results", use_container_width=True)
            
            if hasattr(st.session_state, 'results') and st.session_state.results:
                st.subheader("📈 Gaze Analysis")
                
                for result in st.session_state.results:
                    with st.expander(f"👤 Face {result['face_id']} (Confidence: {result['confidence']:.2f})"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                "🔄 Pitch (Vertical)", 
                                f"{result['pitch_degrees']:.1f}°",
                                help="Negative = Looking down, Positive = Looking up"
                            )
                        
                        with col_b:
                            st.metric(
                                "↔️ Yaw (Horizontal)", 
                                f"{result['yaw_degrees']:.1f}°",
                                help="Negative = Looking left, Positive = Looking right"
                            )
                        
                        # Gaze direction interpretation
                        pitch = result['pitch_degrees']
                        yaw = result['yaw_degrees']
                        
                        direction = "Looking "
                        if abs(pitch) < 5 and abs(yaw) < 5:
                            direction += "straight ahead 👀"
                        else:
                            if pitch > 10:
                                direction += "up "
                            elif pitch < -10:
                                direction += "down "
                            
                            if yaw > 10:
                                direction += "right ➡️"
                            elif yaw < -10:
                                direction += "left ⬅️"
                        
                        st.success(f"**Gaze Direction:** {direction}")
        else:
            st.info("👆 Upload an image and click 'Estimate Gaze' to see results here")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This Demo")
    st.markdown("""
    This application uses **MobileGaze** for real-time gaze estimation:
    
    **🔧 Technical Stack:**
    - **Model**: MobileOne S0 (4.8MB)
    - **Dataset**: Trained on Gaze360
    - **Face Detection**: OpenCV Haar Cascades
    - **Framework**: PyTorch
    
    **📐 Gaze Measurements:**
    - **Pitch**: Vertical gaze (-180° to +180°)
    - **Yaw**: Horizontal gaze (-180° to +180°)
    """)
    
    st.header("🚀 Performance")
    device_info = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU"
    st.metric("🖥️ Device", device_info)
    st.metric("📦 Model Size", "4.8 MB")
    st.metric("🖼️ Input Resolution", "448x448")
    
    st.header("📚 Learn More")
    st.markdown("""
    - [📄 MobileGaze Paper](https://doi.org/10.5281/zenodo.14257640)
    - [📊 Gaze360 Dataset](https://gaze360.csail.mit.edu/)
    """)
    
    # Add a sample images section
    st.header("🖼️ Try Sample Images")
    sample_images = [f"demo{i}.jpg" for i in range(1, 9) if os.path.exists(f"demo{i}.jpg")]
    
    if sample_images:
        selected_sample = st.selectbox("Choose a sample image:", ["None"] + sample_images)
        if selected_sample != "None" and st.button("Load Sample"):
            st.session_state.sample_image = selected_sample