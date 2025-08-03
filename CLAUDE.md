# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a gaze detection and estimation project that combines multiple approaches:

1. **Moondream2-based gaze detection** - Uses the vikhyatk/moondream2 model for face detection and gaze estimation
2. **MobileGaze neural networks** - Custom implementation using MobileOne, ResNet, and MobileNet architectures
3. **Multiple demo applications** - Various interfaces including Gradio, Streamlit, and simple Python scripts

## Development Commands

### Installation
```bash
# Install main dependencies
pip install -r requirements.txt

# Install eye-direction-predictor dependencies
pip install -r eye-direction-predictor/requirements.txt
```

### Running Applications
```bash
# Gradio demo (main HuggingFace Space app)
python demo.py

# Streamlit variations
streamlit run streamlit_app.py
streamlit run app.py

# Simple demos
python simple_demo.py
python simple_app.py
python portfolio_app.py
python new_app.py
```

### Eye Direction Predictor (MobileGaze)
```bash
cd eye-direction-predictor

# Download model weights
sh download.sh [model_name]  # resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0

# Training
python main.py --data [dataset_path] --dataset [dataset_name] --arch [architecture_name]

# Evaluation
python evaluate.py --data [dataset_path] --dataset [dataset_name] --weight [weight_path] --arch [architecture_name]

# Inference
python inference.py --model [model_name] --weight [model_weight_path] --source [source_video/cam_index] --output [output_file]

# ONNX export and inference
python onnx_export.py --weight [model_path] --model [model_name] --dynamic
python onnx_inference.py --source [source] --model [onnx_model_path] --output [output_path]
```

## Architecture

### Core Components

1. **demo.py** - Main Gradio application using Moondream2 for gaze detection
2. **gaze_app/** - Modular gaze estimation package with MobileGaze models
3. **eye-direction-predictor/** - Complete gaze estimation framework with training/inference

### Model Implementations

**Moondream2 Integration (demo.py, streamlit_app.py)**:
- Face detection using `model.detect(enc_image, "face")`
- Gaze estimation using `model.detect_gaze(enc_image, face=face, eye=face_center)`
- Supports ensemble mode with image flipping for accuracy

**MobileGaze Models (gaze_app/, eye-direction-predictor/)**:
- ResNet variants (18, 34, 50)
- MobileNet v2
- MobileOne (s0-s4) - optimized for mobile inference
- Trained on Gaze360 and MPIIGaze datasets

### Key Configuration

**Dataset Configurations (gaze_app/config.py)**:
- Gaze360: 90 bins, 4° binwidth, ±180° range
- MPIIGaze: 28 bins, 3° binwidth, ±42° range

**Model Loading**:
- Use `get_model(arch, bins, inference_mode=True)` from `gaze_app.utils.helpers`
- Models located in `gaze_app/weights/` or `eye-direction-predictor/weights/`

### Device Support

- **MPS (Apple Silicon)**: Primary target with `torch.backends.mps.is_available()`
- **CPU fallback**: Automatic detection and fallback
- All models use `torch.float32` for MPS compatibility

### Data Processing Pipeline

1. **Face Detection**: OpenCV Haar Cascades or Moondream2
2. **Preprocessing**: 448x448 resize, ImageNet normalization
3. **Gaze Prediction**: Softmax classification to angular bins
4. **Visualization**: Colored arrows and bounding boxes with matplotlib/OpenCV

## File Structure Notes

- **Demo images**: demo1.jpg through demo8.jpg for testing
- **Dual implementations**: Both `gaze_app/` and `eye-direction-predictor/` contain similar model architectures
- **Multiple UIs**: Various app files provide different interfaces to the same core functionality
- **Weight management**: Models require separate download via download.sh script

## Important Notes

- Moondream2 model requires `trust_remote_code=True`
- MobileGaze models support both training and inference modes
- ONNX export available for deployment optimization
- All visualization uses non-interactive matplotlib backend ("Agg")