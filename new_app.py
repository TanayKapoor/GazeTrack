import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import sys
import os
import cv2
import numpy as np

# Fix PyTorch-Streamlit compatibility issue
import torch
torch.classes.__path__ = []

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

try:
    import ultralytics
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Add gaze_app to path  
gaze_app_path = os.path.join(os.getcwd(), 'gaze_app')
sys.path.insert(0, gaze_app_path)

try:
    from config import data_config
    from utils.helpers import get_model
    import uniface
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Make sure gaze_app directory exists and has the required files")
    st.stop()

# Streamlit configuration
st.set_page_config(page_title="Gaze Estimation Demo", layout="wide")
st.title("🎯 Gaze Estimation with MobileGaze")
st.markdown("Advanced gaze direction estimation using lightweight deep learning models")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.face_detector = None
    st.session_state.gaze_detector = None

@st.cache_resource
def load_models():
    """Load face detection, gaze estimation, and object detection models"""
    try:
        # Initialize face detector
        face_detector = uniface.RetinaFace(model="retinaface_r34")
        
        # Model configuration for gaze360 dataset
        dataset_config = data_config["gaze360"]
        bins = dataset_config["bins"]
        binwidth = dataset_config["binwidth"] 
        angle = dataset_config["angle"]
        
        # Device setup
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
        
        # Load gaze estimation model
        gaze_detector = get_model("mobileone_s0", bins, inference_mode=True)
        state_dict = torch.load("./gaze_app/weights/mobileone_s0.pt", map_location=device)
        gaze_detector.load_state_dict(state_dict)
        gaze_detector.to(device)
        gaze_detector.eval()
        
        # Load object detection model if available
        object_detector = None
        if YOLO_AVAILABLE:
            try:
                from ultralytics import YOLO
                object_detector = YOLO('yolov8n.pt')  # Lightweight model
            except Exception as e:
                st.warning(f"Could not load YOLO model: {e}")
        
        return face_detector, gaze_detector, object_detector, device, idx_tensor, binwidth, angle
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
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

def get_confidence_colors(confidence):
    """Generate colors based on confidence level"""
    # High confidence: bright green to cyan
    # Medium confidence: yellow to orange  
    # Low confidence: red to dark red
    if confidence > 0.8:
        # High confidence - bright cyan/green
        primary = (0, 255, 200)
        secondary = (100, 255, 255)
    elif confidence > 0.6:
        # Medium-high confidence - cyan/blue
        primary = (255, 200, 0)
        secondary = (255, 255, 100)
    elif confidence > 0.4:
        # Medium confidence - yellow/orange
        primary = (0, 165, 255)
        secondary = (100, 200, 255)
    else:
        # Low confidence - red
        primary = (0, 100, 255)
        secondary = (50, 150, 255)
    
    return primary, secondary

def draw_gradient_arrow(image, start, end, primary_color, secondary_color, thickness, confidence):
    """Draw arrow with gradient effect based on confidence"""
    # Number of segments for gradient
    segments = 20
    
    # Calculate step vectors
    dx = (end[0] - start[0]) / segments
    dy = (end[1] - start[1]) / segments
    
    # Draw gradient segments
    for i in range(segments):
        # Calculate alpha for gradient (stronger at start)
        alpha = 1.0 - (i / segments) * (1.0 - confidence)
        
        # Calculate current segment points
        x1 = int(start[0] + dx * i)
        y1 = int(start[1] + dy * i)
        x2 = int(start[0] + dx * (i + 1))
        y2 = int(start[1] + dy * (i + 1))
        
        # Interpolate colors
        seg_color = tuple(int(primary_color[j] * alpha + secondary_color[j] * (1 - alpha)) for j in range(3))
        seg_thickness = max(1, int(thickness * alpha))
        
        cv2.line(image, (x1, y1), (x2, y2), seg_color, seg_thickness, cv2.LINE_AA)

def draw_gaze_on_image(image, bbox, pitch, yaw, confidence=1.0, thickness=8):
    """Draw enhanced gaze direction visualization with confidence-based gradient"""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    
    # Get confidence-based colors
    primary_color, secondary_color = get_confidence_colors(confidence)
    
    # Calculate center of face
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    # Calculate gaze direction with much longer arrow (2.5x length)
    base_length = max(x_max - x_min, 150)
    length = int(base_length * 2.5 * confidence)  # Confidence affects length
    dx = int(-length * np.sin(pitch) * np.cos(yaw))
    dy = int(-length * np.sin(yaw))
    
    point1 = (x_center, y_center)
    point2 = (x_center + dx, y_center + dy)
    
    # Draw enhanced face bounding box with confidence-based intensity
    box_thickness = max(2, int(4 * confidence))
    confidence_alpha = int(255 * confidence)
    
    # Multiple glow layers for depth
    for glow in range(3, 0, -1):
        glow_alpha = confidence_alpha // (glow + 1)
        glow_color = tuple(min(255, int(c * glow_alpha / 255)) for c in primary_color)
        cv2.rectangle(image, (x_min-glow, y_min-glow), (x_max+glow, y_max+glow), 
                     glow_color, 1)
    
    # Main bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), primary_color, box_thickness)
    
    # Draw multiple glow layers for arrow
    base_thickness = max(6, int(thickness * confidence))
    
    # Outer glow (widest, darkest)
    glow_color = tuple(c//3 for c in primary_color)
    cv2.arrowedLine(image, point1, point2, glow_color, base_thickness+8, 
                   line_type=cv2.LINE_AA, tipLength=0.2)
    
    # Middle glow
    mid_glow = tuple(c//2 for c in primary_color)
    cv2.arrowedLine(image, point1, point2, mid_glow, base_thickness+4, 
                   line_type=cv2.LINE_AA, tipLength=0.25)
    
    # Draw gradient arrow body
    draw_gradient_arrow(image, point1, point2, primary_color, secondary_color, base_thickness, confidence)
    
    # Main arrow outline
    cv2.arrowedLine(image, point1, point2, primary_color, base_thickness, 
                   line_type=cv2.LINE_AA, tipLength=0.3)
    
    # Accent arrow (thinner, brighter)
    cv2.arrowedLine(image, point1, point2, secondary_color, max(2, base_thickness-2), 
                   line_type=cv2.LINE_AA, tipLength=0.35)
    
    # Enhanced center point with confidence-based size
    center_size = max(8, int(15 * confidence))
    
    # Multiple concentric circles with confidence effect
    cv2.circle(image, point1, radius=center_size+8, color=glow_color, thickness=3)
    cv2.circle(image, point1, radius=center_size+4, color=primary_color, thickness=3)
    cv2.circle(image, point1, radius=center_size, color=secondary_color, thickness=-1)
    cv2.circle(image, point1, radius=center_size-4, color=primary_color, thickness=-1)
    cv2.circle(image, point1, radius=max(2, center_size-8), color=(255, 255, 255), thickness=-1)
    
    # Enhanced arrow tip with confidence-based size
    arrow_len = np.sqrt(dx*dx + dy*dy)
    if arrow_len > 0:
        dx_norm = dx / arrow_len
        dy_norm = dy / arrow_len
        
        # Larger, more prominent arrow tip
        tip_size = max(15, int(25 * confidence))
        tip1_x = int(point2[0] - tip_size * (dx_norm + dy_norm * 0.6))
        tip1_y = int(point2[1] - tip_size * (dy_norm - dx_norm * 0.6))
        tip2_x = int(point2[0] - tip_size * (dx_norm - dy_norm * 0.6))
        tip2_y = int(point2[1] - tip_size * (dy_norm + dx_norm * 0.6))
        
        # Draw multi-layered arrow tip
        arrow_pts = np.array([point2, (tip1_x, tip1_y), (tip2_x, tip2_y)], np.int32)
        
        # Shadow tip
        shadow_pts = arrow_pts + 2
        cv2.fillPoly(image, [shadow_pts], glow_color)
        
        # Main tip
        cv2.fillPoly(image, [arrow_pts], primary_color)
        cv2.polylines(image, [arrow_pts], True, secondary_color, 3, cv2.LINE_AA)
        
        # Inner highlight
        inner_pts = np.array([point2, 
                             ((tip1_x + point2[0])//2, (tip1_y + point2[1])//2),
                             ((tip2_x + point2[0])//2, (tip2_y + point2[1])//2)], np.int32)
        cv2.fillPoly(image, [inner_pts], (255, 255, 255))
    
    return image

def detect_objects(image, object_detector):
    """Detect objects in the image using YOLO"""
    if object_detector is None:
        return []
    
    try:
        results = object_detector(image, verbose=False)
        objects = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Filter out low confidence detections
                    if conf > 0.3:
                        objects.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'class': cls,
                            'name': object_detector.names[cls],
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
        
        return objects
    except Exception as e:
        st.warning(f"Object detection failed: {e}")
        return []

def find_gaze_target(gaze_point, gaze_direction, objects, image_shape):
    """Find which object the gaze is likely targeting"""
    if not objects:
        return None
    
    gaze_x, gaze_y = gaze_point
    dx, dy = gaze_direction
    
    # Extend gaze line to image boundaries
    img_height, img_width = image_shape[:2]
    
    # Normalize direction
    length = np.sqrt(dx*dx + dy*dy)
    if length == 0:
        return None
    
    dx_norm, dy_norm = dx/length, dy/length
    
    # Find intersections with objects
    best_object = None
    min_distance = float('inf')
    
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        obj_center_x, obj_center_y = obj['center']
        
        # Check if gaze line intersects with object bounding box
        # Simple line-rectangle intersection
        for t in range(0, max(img_width, img_height), 5):  # Sample points along gaze line
            point_x = gaze_x + dx_norm * t
            point_y = gaze_y + dy_norm * t
            
            # Check if point is within object bbox
            if x1 <= point_x <= x2 and y1 <= point_y <= y2:
                # Calculate distance from gaze origin to intersection
                distance = np.sqrt((point_x - gaze_x)**2 + (point_y - gaze_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_object = obj
                break
    
    return best_object

def draw_object_boxes(image, objects, gaze_targets=None):
    """Draw bounding boxes around detected objects"""
    if gaze_targets is None:
        gaze_targets = []
    
    for obj in objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        
        # Different colors for gaze targets vs other objects
        if obj in gaze_targets:
            color = (255, 255, 0)  # Bright yellow for gaze targets
            thickness = 4
        else:
            color = (128, 128, 128)  # Gray for other objects
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label
        label = f"{obj['name']} ({obj['confidence']:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background for text
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 0, 0), 2)
    
    return image

def process_image(uploaded_image, show_confidence=False, detect_objects_flag=False):
    """Process uploaded image for gaze estimation and object detection"""
    
    # Load models
    model_results = load_models()
    if len(model_results) == 7:
        face_detector, gaze_detector, object_detector, device, idx_tensor, binwidth, angle = model_results
    else:
        face_detector, gaze_detector, device, idx_tensor, binwidth, angle = model_results
        object_detector = None
    
    if face_detector is None or gaze_detector is None:
        return None, "Failed to load models"
    
    try:
        # Convert PIL to OpenCV format
        image = np.array(uploaded_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detect objects if requested
        objects = []
        if detect_objects_flag and object_detector is not None:
            objects = detect_objects(image, object_detector)
        
        # Detect faces
        bboxes, keypoints = face_detector.detect(image)
        
        if len(bboxes) == 0:
            return None, "No faces detected in the image"
        
        results = []
        processed_image = image.copy()
        gaze_targets = []
        
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
                
                # Find gaze target if objects are detected
                gaze_target = None
                if objects:
                    face_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
                    gaze_direction = (
                        int(-150 * np.sin(pitch_rad) * np.cos(yaw_rad)),
                        int(-150 * np.sin(yaw_rad))
                    )
                    gaze_target = find_gaze_target(face_center, gaze_direction, objects, image.shape)
                    if gaze_target and gaze_target not in gaze_targets:
                        gaze_targets.append(gaze_target)
                
                # Draw gaze on image with confidence
                processed_image = draw_gaze_on_image(processed_image, bbox, pitch_rad, yaw_rad, confidence)
                
                result_data = {
                    'face_id': i + 1,
                    'confidence': confidence,
                    'pitch_degrees': pitch_angle.cpu().numpy()[0],
                    'yaw_degrees': yaw_angle.cpu().numpy()[0],
                    'pitch_radians': pitch_rad,
                    'yaw_radians': yaw_rad
                }
                
                if gaze_target:
                    result_data['gaze_target'] = gaze_target['name']
                    result_data['target_confidence'] = gaze_target['confidence']
                
                results.append(result_data)
        
        # Draw object boxes if objects were detected
        if objects:
            processed_image = draw_object_boxes(processed_image, objects, gaze_targets)
        
        # Convert back to RGB for display
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image, results
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Main Streamlit interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing faces for gaze estimation"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Processing options
        st.subheader("⚙️ Options")
        show_confidence = st.checkbox("Show detection confidence", value=True)
        detect_objects = st.checkbox("🎯 Detect objects and link gaze", value=True, 
                                   help="Enable object detection to see what people are looking at")
        
        if not YOLO_AVAILABLE and detect_objects:
            st.warning("⚠️ Object detection requires ultralytics package. Install with: `pip install ultralytics`")
        
        if st.button("🎯 Estimate Gaze", type="primary"):
            with st.spinner("Processing image..."):
                processed_image, results = process_image(image, show_confidence, detect_objects and YOLO_AVAILABLE)
                
                if processed_image is not None:
                    st.session_state.processed_image = processed_image
                    st.session_state.results = results
                else:
                    st.error(results)

with col2:
    st.header("📊 Results")
    
    if hasattr(st.session_state, 'processed_image') and st.session_state.processed_image is not None:
        # Display processed image
        st.image(st.session_state.processed_image, caption="Gaze Estimation Results", use_container_width=True)
        
        # Display results table
        if hasattr(st.session_state, 'results') and st.session_state.results:
            st.subheader("📈 Detection Details")
            
            for result in st.session_state.results:
                gaze_info = ""
                if 'gaze_target' in result:
                    gaze_info = f" → 👁️ {result['gaze_target']}"
                
                with st.expander(f"Face {result['face_id']} (Confidence: {result['confidence']:.2f}){gaze_info}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Pitch (Vertical)", f"{result['pitch_degrees']:.1f}°")
                        st.metric("Yaw (Horizontal)", f"{result['yaw_degrees']:.1f}°")
                        
                        if 'gaze_target' in result:
                            st.success(f"🎯 Looking at: **{result['gaze_target']}** (conf: {result['target_confidence']:.2f})")
                    
                    with col_b:
                        st.metric("Pitch (Radians)", f"{result['pitch_radians']:.3f}")
                        st.metric("Yaw (Radians)", f"{result['yaw_radians']:.3f}")
    else:
        st.info("👆 Upload an image and click 'Estimate Gaze' to see results")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses **MobileGaze** for real-time gaze estimation:
    
    - **Model**: MobileOne S0 (4.8MB)
    - **Dataset**: Trained on Gaze360
    - **Face Detection**: RetinaFace (uniface)
    - **Object Detection**: YOLOv8n (optional)
    - **Framework**: PyTorch
    
    **Features:**
    - **Confidence-based visualization**: Arrow brightness/length varies with detection confidence
    - **Object linking**: Identifies what people are looking at
    - **Enhanced arrows**: Multi-layered gradient arrows with glow effects
    
    **Gaze Angles:**
    - **Pitch**: Vertical gaze direction (-42° to +42°)
    - **Yaw**: Horizontal gaze direction (-42° to +42°)
    """)
    
    st.header("🔧 Technical Details")
    device_info = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU"
    st.metric("Device", device_info)
    st.metric("Model Size", "4.8 MB")
    st.metric("Input Resolution", "448x448")
    
    st.header("📚 References")
    st.markdown("""
    - [MobileGaze Paper](https://doi.org/10.5281/zenodo.14257640)
    - [Gaze360 Dataset](https://gaze360.csail.mit.edu/)
    """)