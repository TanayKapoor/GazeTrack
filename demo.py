import gradio as gr
import os
import torch

# Add your HF token here if not logged in via CLI
# os.environ['HF_TOKEN'] = 'hf_your_token_here'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image
from transformers import AutoModelForCausalLM

matplotlib.use("Agg")  # Use Agg backend for non-interactive plotting

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=device
)
model.to(device)

def visualize_faces_and_gaze(face_boxes, gaze_points=None, image=None, show_plot=True):
    """Visualization function that can handle faces without gaze data"""
    # Calculate figure size based on image aspect ratio
    if image is not None:
        height, width = image.shape[:2]
        aspect_ratio = width / height
        fig_height = 6  # Base height
        fig_width = fig_height * aspect_ratio
    else:
        width, height = 800, 600
        fig_width, fig_height = 10, 8

    # Create figure with tight layout
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("#1a1a1a")
        fig.patch.set_facecolor("#1a1a1a")

    colors = plt.cm.rainbow(np.linspace(0, 1, len(face_boxes)))

    for i, (face_box, color) in enumerate(zip(face_boxes, colors)):
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )

        x, y, width_box, height_box = face_box
        face_center_x = x + width_box / 2
        face_center_y = y + height_box / 2

        # Draw face bounding box
        face_rect = plt.Rectangle(
            (x, y), width_box, height_box, fill=False, color=hex_color, linewidth=2
        )
        ax.add_patch(face_rect)

        # Draw gaze line if gaze data is available
        if gaze_points is not None and i < len(gaze_points) and gaze_points[i] is not None:
            gaze_x, gaze_y = gaze_points[i]
            
            points = 50
            alphas = np.linspace(0.8, 0, points)

            x_points = np.linspace(face_center_x, gaze_x, points)
            y_points = np.linspace(face_center_y, gaze_y, points)

            for j in range(points - 1):
                ax.plot(
                    [x_points[j], x_points[j + 1]],
                    [y_points[j], y_points[j + 1]],
                    color=hex_color,
                    alpha=alphas[j],
                    linewidth=4,
                )

            ax.scatter(gaze_x, gaze_y, color=hex_color, s=100, zorder=5)
            ax.scatter(gaze_x, gaze_y, color="white", s=50, zorder=6)

    # Set plot limits and remove axes
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove padding around the plot
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig

def process_image(input_image, use_ensemble):
    if input_image is None:
        return None, ""
        
    try:
        # Convert to PIL Image if needed
        if isinstance(input_image, np.ndarray):
            pil_image = Image.fromarray(input_image)
        else:
            pil_image = input_image

        # Get image encoding
        enc_image = model.encode_image(pil_image)
        if use_ensemble:
            flipped_pil = pil_image.copy().transpose(method=Image.FLIP_LEFT_RIGHT)
            flip_enc_image = model.encode_image(flipped_pil)
        else:
            flip_enc_image = None

        # Detect faces
        faces = model.detect(enc_image, "face")["objects"]

        if not faces:
            return None, "No faces detected in the image."

        # Process each face
        face_boxes = []
        gaze_points = []

        for face in faces:
            # Add face bounding box regardless of gaze detection
            face_box = (
                face["x_min"] * pil_image.width,
                face["y_min"] * pil_image.height,
                (face["x_max"] - face["x_min"]) * pil_image.width,
                (face["y_max"] - face["y_min"]) * pil_image.height,
            )
            face_center = (
                (face["x_min"] + face["x_max"]) / 2,
                (face["y_min"] + face["y_max"]) / 2
            )
            face_boxes.append(face_box)

            # Try to detect gaze
            gaze_settings = {
                "prioritize_accuracy": use_ensemble,
                "flip_enc_img": flip_enc_image
            }
            gaze = model.detect_gaze(enc_image, face=face, eye=face_center, unstable_settings=gaze_settings)["gaze"]

            if gaze is not None:
                gaze_point = (
                    gaze["x"] * pil_image.width,
                    gaze["y"] * pil_image.height,
                )
                gaze_points.append(gaze_point)
            else:
                gaze_points.append(None)

        # Create visualization
        image_array = np.array(pil_image)
        fig = visualize_faces_and_gaze(
            face_boxes, gaze_points, image=image_array, show_plot=False
        )

        faces_with_gaze = sum(1 for gp in gaze_points if gp is not None)
        status = f"Found {len(faces)} faces. {len(faces) - faces_with_gaze} gazing out of frame."
        return fig, status

    except Exception as e:
        return None, f"Error processing image: {str(e)}"

with gr.Blocks(title="Moondream Gaze Detection") as app:
    gr.Markdown("# 🌔 Moondream Gaze Detection")
    gr.Markdown("Upload an image to detect faces and visualize their gaze directions. Join the [Moondream Discord server](https://discord.com/invite/tRUdpjDQfH) if you have questions about how this works.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            use_ensemble = gr.Checkbox(
                label="Use Ensemble Mode", 
                value=False,
                info="Ensemble mode combines multiple predictions for higher accuracy."
            )

        with gr.Column():
            output_text = gr.Textbox(label="Status")
            output_plot = gr.Plot(label="Visualization")

    input_image.change(
        fn=process_image, inputs=[input_image, use_ensemble], outputs=[output_plot, output_text]
    )
    use_ensemble.change(
        fn=process_image, inputs=[input_image, use_ensemble], outputs=[output_plot, output_text]
    )

    gr.Examples(
        examples=["demo1.jpg", "demo2.jpg", "demo3.jpg", "demo4.jpg", "demo5.jpg", "demo6.jpg", "demo7.jpg", "demo8.jpg"],
        inputs=input_image,
    )

if __name__ == "__main__":
    app.launch()