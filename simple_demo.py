#!/usr/bin/env python3
"""
Simple Eye Direction Predictor Demo
A minimal version that works without Streamlit
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
import os

# Add gaze_app to path
sys.path.insert(0, os.path.join(os.getcwd(), 'gaze_app'))

try:
    from config import data_config
    from utils.helpers import get_model
    import uniface
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure gaze_app directory exists with required files")
    sys.exit(1)

class EyeDirectionPredictor:
    def __init__(self):
        self.face_detector = None
        self.gaze_detector = None
        self.device = None
        self.idx_tensor = None
        self.binwidth = None
        self.angle = None
        self.load_models()
    
    def load_models(self):
        """Load face detection and gaze estimation models"""
        try:
            print("🔄 Loading models...")
            
            # Initialize face detector
            self.face_detector = uniface.RetinaFace(model="retinaface_r34")
            print("✅ Face detector loaded")
            
            # Model configuration
            dataset_config = data_config["gaze360"]
            bins = dataset_config["bins"]
            self.binwidth = dataset_config["binwidth"] 
            self.angle = dataset_config["angle"]
            
            # Device setup
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.idx_tensor = torch.arange(bins, device=self.device, dtype=torch.float32)
            print(f"✅ Using device: {self.device}")
            
            # Load gaze estimation model
            self.gaze_detector = get_model("mobileone_s0", bins, inference_mode=True)
            state_dict = torch.load("./gaze_app/weights/mobileone_s0.pt", map_location=self.device)
            self.gaze_detector.load_state_dict(state_dict)
            self.gaze_detector.to(self.device)
            self.gaze_detector.eval()
            print("✅ Gaze estimation model loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            sys.exit(1)
    
    def preprocess_face(self, face_image):
        """Preprocess face image for prediction"""
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(face_image).unsqueeze(0)
    
    def draw_predictions(self, image, bbox, pitch, yaw, confidence=1.0):
        """Draw prediction overlays on image"""
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        
        # Face center
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        
        # Calculate gaze direction
        length = max(x_max - x_min, 100)
        dx = int(-length * 1.5 * np.sin(pitch) * np.cos(yaw))
        dy = int(-length * 1.5 * np.sin(yaw))
        
        # Colors based on confidence
        color = (0, 255, 0) if confidence > 0.8 else (255, 165, 0) if confidence > 0.5 else (255, 0, 0)
        
        # Draw face box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Draw gaze vector
        end_point = (x_center + dx, y_center + dy)
        cv2.arrowedLine(image, (x_center, y_center), end_point, color, 4, tipLength=0.3)
        
        # Draw center point
        cv2.circle(image, (x_center, y_center), 8, color, -1)
        cv2.circle(image, (x_center, y_center), 4, (255, 255, 255), -1)
        
        # Add text
        cv2.putText(image, f"Conf: {confidence:.2f}", (x_min, y_min-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Pitch: {np.degrees(pitch):.1f}°", (x_min, y_max+25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, f"Yaw: {np.degrees(yaw):.1f}°", (x_min, y_max+50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def process_image(self, image_path):
        """Process image for eye direction prediction"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
            
            print(f"🔄 Processing image: {image_path}")
            
            # Detect faces
            bboxes, keypoints = self.face_detector.detect(image)
            
            if len(bboxes) == 0:
                print("❌ No faces detected in the image")
                return None
            
            print(f"✅ Detected {len(bboxes)} face(s)")
            
            processed_image = image.copy()
            
            # Process each face
            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                confidence = bbox[4] if len(bbox) > 4 else 1.0
                
                print(f"📍 Processing face {i+1} (confidence: {confidence:.2f})")
                
                # Extract and preprocess face
                face_region = image[y_min:y_max, x_min:x_max]
                if face_region.size == 0:
                    continue
                    
                face_tensor = self.preprocess_face(face_region).to(self.device)
                
                # Predict gaze
                with torch.no_grad():
                    pitch_pred, yaw_pred = self.gaze_detector(face_tensor)
                    
                    pitch_prob = F.softmax(pitch_pred, dim=1)
                    yaw_prob = F.softmax(yaw_pred, dim=1)
                    
                    pitch_angle = torch.sum(pitch_prob * self.idx_tensor, dim=1) * self.binwidth - self.angle
                    yaw_angle = torch.sum(yaw_prob * self.idx_tensor, dim=1) * self.binwidth - self.angle
                    
                    pitch_rad = np.radians(pitch_angle.cpu().numpy()[0])
                    yaw_rad = np.radians(yaw_angle.cpu().numpy()[0])
                    
                    print(f"   👁️ Pitch: {pitch_angle.cpu().numpy()[0]:.1f}° | Yaw: {yaw_angle.cpu().numpy()[0]:.1f}°")
                    
                    # Draw predictions
                    processed_image = self.draw_predictions(processed_image, bbox, pitch_rad, yaw_rad, confidence)
            
            return processed_image
            
        except Exception as e:
            print(f"❌ Error processing image: {str(e)}")
            return None
    
    def run_demo(self):
        """Run interactive demo"""
        print("\n" + "="*60)
        print("👁️  EYE DIRECTION PREDICTOR DEMO")
        print("="*60)
        print("\nCommands:")
        print("  - Enter image path to process")
        print("  - 'webcam' for live webcam demo")
        print("  - 'quit' to exit")
        print("\n" + "="*60)
        
        while True:
            try:
                user_input = input("\n💬 Enter command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'webcam':
                    self.run_webcam_demo()
                
                elif user_input:
                    if os.path.exists(user_input):
                        result = self.process_image(user_input)
                        if result is not None:
                            # Save result
                            output_path = f"output_{os.path.basename(user_input)}"
                            cv2.imwrite(output_path, result)
                            print(f"✅ Result saved as: {output_path}")
                            
                            # Display result
                            cv2.imshow('Eye Direction Prediction', result)
                            print("👀 Press any key to close the image window...")
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                    else:
                        print(f"❌ File not found: {user_input}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def run_webcam_demo(self):
        """Run live webcam demo"""
        print("🔄 Starting webcam... Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                bboxes, _ = self.face_detector.detect(frame)
                
                if len(bboxes) > 0:
                    for bbox in bboxes:
                        x_min, y_min, x_max, y_max = map(int, bbox[:4])
                        confidence = bbox[4] if len(bbox) > 4 else 1.0
                        
                        # Quick face processing
                        face_region = frame[y_min:y_max, x_min:x_max]
                        if face_region.size > 0:
                            face_tensor = self.preprocess_face(face_region).to(self.device)
                            
                            with torch.no_grad():
                                pitch_pred, yaw_pred = self.gaze_detector(face_tensor)
                                pitch_prob = F.softmax(pitch_pred, dim=1)
                                yaw_prob = F.softmax(yaw_pred, dim=1)
                                
                                pitch_angle = torch.sum(pitch_prob * self.idx_tensor, dim=1) * self.binwidth - self.angle
                                yaw_angle = torch.sum(yaw_prob * self.idx_tensor, dim=1) * self.binwidth - self.angle
                                
                                pitch_rad = np.radians(pitch_angle.cpu().numpy()[0])
                                yaw_rad = np.radians(yaw_angle.cpu().numpy()[0])
                                
                                frame = self.draw_predictions(frame, bbox, pitch_rad, yaw_rad, confidence)
                
                cv2.imshow('Live Eye Direction Prediction', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"❌ Webcam error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    print("🚀 Initializing Eye Direction Predictor...")
    
    # Check if demo images exist
    demo_images = [f"demo{i}.jpg" for i in range(1, 9) if os.path.exists(f"demo{i}.jpg")]
    
    if demo_images:
        print(f"\n📸 Found demo images: {', '.join(demo_images)}")
    
    # Initialize predictor
    predictor = EyeDirectionPredictor()
    
    # Run demo
    predictor.run_demo()

if __name__ == "__main__":
    main()