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

# Add gaze_app to path
current_dir = os.path.dirname(os.path.abspath(__file__))
gaze_dir = os.path.join(current_dir, 'gaze_app')
sys.path.insert(0, gaze_dir)

# Streamlit configuration
st.set_page_config(page_title="Gaze Estimation Demo", layout="wide")

@st.cache_resource
def load_models():
    """Load face detection and gaze estimation models"""
    try:
        # Import here to avoid issues
        from config import data_config
        from utils.helpers import get_model
        import uniface
        
        # Initialize face detector
        try:
            face_detector = uniface.RetinaFace('retinaface_r50_v1')
        except:
            try:
                face_detector = uniface.RetinaFace('retinaface_mobile_v1')
            except:
                # Use OpenCV as fallback
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                face_detector = None
        
        # Model configuration for gaze360 dataset
        dataset_config = data_config["gaze360"]
        bins = dataset_config["bins"]
        binwidth = dataset_config["binwidth"] 
        angle = dataset_config["angle"]
        
        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
        
        # Load gaze estimation model
        weight_path = os.path.join(gaze_dir, "weights", "mobileone_s0.pt")
        if not os.path.exists(weight_path):
            st.error(f"Model weights not found at {weight_path}")
            return None, None, None, None, None, None, None
            
        gaze_detector = get_model("mobileone_s0", bins, inference_mode=True)
        state_dict = torch.load(weight_path, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        gaze_detector.to(device)
        gaze_detector.eval()
        
        st.success("✅ Models loaded successfully!")
        return face_detector, gaze_detector, device, idx_tensor, binwidth, angle, face_cascade if face_detector is None else None
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.error("Make sure gaze_app directory and weights exist")
        return None, None, None, None, None, None, None

def preprocess_face(face_image):
    """Preprocess face image for gaze estimation"""
    # Convert BGR to RGB if needed
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
    length = x_max - x_min
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))
    
    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)
    
    # Draw face bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    
    # Draw gaze direction
    cv2.circle(image, point1, radius=5, color=color, thickness=-1)
    cv2.arrowedLine(image, point1, point2, color=color, thickness=thickness, 
                   line_type=cv2.LINE_AA, tipLength=0.3)
    
    return image

def process_image(uploaded_image):
    """Process uploaded image for gaze estimation"""
    
    # Load models
    models = load_models()
    if models[0] is None:
        return None, "Failed to load models"
    
    face_detector, gaze_detector, device, idx_tensor, binwidth, angle, face_cascade = models
    
    try:
        # Convert PIL to OpenCV format
        image = np.array(uploaded_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        if face_detector is not None:
            # Use RetinaFace
            bboxes, keypoints = face_detector.detect(image)
        else:
            # Use OpenCV as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            bboxes = []
            keypoints = []
            for (x, y, w, h) in faces:
                # Convert to RetinaFace format [x_min, y_min, x_max, y_max, confidence]
                bboxes.append([x, y, x+w, y+h, 0.9])
                keypoints.append(None)
        
        if len(bboxes) == 0:
            return None, "No faces detected in the image"
        
        results = []
        processed_image = image.copy()
        
        # Process each detected face
        for i, (bbox, keypoint) in enumerate(zip(bboxes, keypoints)):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            confidence = bbox[4] if len(bbox) > 4 else 1.0
            
            # Extract face region
            face_region = image[y_min:y_max, x_min:x_max]
            
            if face_region.size == 0:
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
                
                # Convert to radians
                pitch_rad = np.radians(pitch_angle.cpu().numpy()[0])
                yaw_rad = np.radians(yaw_angle.cpu().numpy()[0])
                
                # Draw gaze on image
                processed_image = draw_gaze_on_image(processed_image, bbox, pitch_rad, yaw_rad)
                
                results.append({
                    'face_id': i + 1,
                    'confidence': confidence,
                    'pitch_degrees': pitch_angle.cpu().numpy()[0],
                    'yaw_degrees': yaw_angle.cpu().numpy()[0],
                })
        
        # Convert back to RGB for display
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image, results
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Main UI
st.title("🎯 Gaze Estimation Demo")
st.markdown("Upload an image to detect faces and estimate gaze direction")

# Check if gaze_app directory exists
if not os.path.exists(gaze_dir):
    st.error("❌ gaze_app directory not found!")
    st.error("Make sure you're running from the correct directory")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(image, use_container_width=True)
        
        if st.button("🎯 Estimate Gaze", type="primary"):
            with st.spinner("Processing..."):
                processed_image, results = process_image(image)
                
                if processed_image is not None:
                    st.session_state.processed_image = processed_image
                    st.session_state.results = results
                    st.success("✅ Processing complete!")
                else:
                    st.error(f"❌ {results}")
    
    with col2:
        st.subheader("📊 Results")
        
        if hasattr(st.session_state, 'processed_image') and st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, use_container_width=True)
            
            if hasattr(st.session_state, 'results') and st.session_state.results:
                st.subheader("📈 Gaze Angles")
                
                for result in st.session_state.results:
                    st.write(f"**Face {result['face_id']}** (Confidence: {result['confidence']:.2f})")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Pitch", f"{result['pitch_degrees']:.1f}°")
                    with col_b:
                        st.metric("Yaw", f"{result['yaw_degrees']:.1f}°")
        else:
            st.info("👆 Upload an image and click 'Estimate Gaze'")

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **MobileGaze Demo**
    - Model: MobileOne S0 (4.8MB)
    - Dataset: Gaze360
    - Face Detection: RetinaFace
    """)
    
    device_info = "MPS" if torch.backends.mps.is_available() else "CPU"
    st.metric("Device", device_info)