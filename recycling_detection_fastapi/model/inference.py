import sys
import os
import torch
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase # Make sure rfdetr is installed in your environment

# --- Configuration ---
NUM_CLASSES = 2 # Changed to 2 classes
MODEL_PATH = os.environ.get("MODEL_PATH", "./loopvision_2025Jun25.pth") # Default to current dir, can be overridden by env var
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5)) # Default to 0.5, can be overridden
CLASS_NAMES = ["CANS", "PET"] # Updated class names

# Global variable to hold the loaded model
model = None

def load_model():
    """
    Loads the RFDETRBase model. This function should be called once at startup.
    """
    global model
    if model is None:
        print(f"Attempting to load RFDETRBase model from: {MODEL_PATH}")
        try:
            # RFDETRBase loads the .pth weights internally via pretrain_weights
            # Assuming MODEL_PATH should be the full path including .pth
            model = RFDETRBase(
                num_classes=NUM_CLASSES,
                pretrain_weights=MODEL_PATH # Ensure this is the correct path to your .pth file
            )
            # RFDETRBase handles moving the model to GPU if CUDA is available,
            # but you might want to explicitly set it or check.
            # if torch.cuda.is_available():
            #     model.to('cuda')
            print("RFDETRBase model loaded successfully.")
        except Exception as e:
            print(f"Error loading RFDETRBase model: {e}")
            sys.exit(1)
    return model

def predict_image(image_path: str):
    """
    Performs inference on a single image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        sv.Detections: The detected objects.
    """
    global model
    if model is None:
        model = load_model()

    if not os.path.exists(image_path):
        print(f"Error: Input image file '{image_path}' not found.")
        return None

    try:
        image = Image.open(image_path).convert("RGB") # Ensure RGB for consistency
    except Exception as e:
        print(f"Error opening or converting image '{image_path}': {e}")
        return None

    print(f"Performing inference on {image_path}...")
    # Make model detections
    detections = model.predict(image, threshold=CONFIDENCE_THRESHOLD)
    print(f"Detected {len(detections)} objects.")
    return detections

def visualize_detections(image_path: str, detections: sv.Detections):
    """
    Visualizes detections on an image. For debugging or demonstration.
    Args:
        image_path (str): Path to the original image.
        detections (sv.Detections): Detections object.
    Returns:
        PIL.Image: Image with annotations.
    """
    if not os.path.exists(image_path):
        print(f"Error: Original image for visualization '{image_path}' not found.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening or converting image '{image_path}' for visualization: {e}")
        return None

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)

    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_thickness=thickness,
        smart_position=True
    )

    detections_labels = []
    # Use the globally defined CLASS_NAMES
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        label = f"{CLASS_NAMES[class_id]} {confidence:.2f}" if class_id < len(CLASS_NAMES) else f"Unknown {confidence:.2f}"
        detections_labels.append(label)

    detections_image = image.copy()
    detections_image = bbox_annotator.annotate(detections_image, detections)
    detections_image = label_annotator.annotate(detections_image, detections, detections_labels)

    return detections_image


if __name__ == "__main__":
    # Check if CLASS_NAMES list length matches NUM_CLASSES
    if len(CLASS_NAMES) != NUM_CLASSES:
        print(f"Warning: CLASS_NAMES list length ({len(CLASS_NAMES)}) does not match NUM_CLASSES ({NUM_CLASSES}). Please verify your configuration.")


    # Load the model once when the script starts
    load_model()

    # How to run inference:
    # Option 1: Provide an image path as a command-line argument
    if len(sys.argv) > 1:
        image_to_predict_path = sys.argv[1]
        # CHANGE HERE: Define the output path INSIDE the container's mounted directory
        # We will mount a host directory to /app/output in the Docker command
        output_dir_in_container = "/app/output"
        output_image_filename = os.path.basename(image_to_predict_path).replace('.', '_detections.') # e.g., test_pet1_detections.png

        # Allow user to specify a custom output filename as 3rd arg, otherwise use generated name
        if len(sys.argv) > 2:
            custom_output_filename = sys.argv[2]
            # Ensure custom_output_filename is just a filename, not a path
            if "/" in custom_output_filename or "\\" in custom_output_filename:
                print("Warning: Custom output filename should not contain directory separators. Using base name.")
                custom_output_filename = os.path.basename(custom_output_filename)
            output_image_filename = custom_output_filename

        output_image_path = os.path.join(output_dir_in_container, output_image_filename)

        # Ensure the output directory exists inside the container (it will if mounted)
        os.makedirs(output_dir_in_container, exist_ok=True)


        detections = predict_image(image_to_predict_path)

        if detections:
            print(f"Detections found: {detections}")

            annotated_image = visualize_detections(image_to_predict_path, detections)
            if annotated_image:
                annotated_image.save(output_image_path)
                print(f"Annotated image saved to: {output_image_path}")
    else:
        print("Usage: python inference.py <path_to_input_image.jpg> [output_filename.jpg]")
        print("\nExample: To test with a dummy image (ensure 'test_image.jpg' exists):")
        print("python inference.py test_image.jpg")
        print("This will save output to output_images/test_image_detections.jpg on your host.")
        print("\nTo specify a custom output filename:")
        print("python inference.py test_image.jpg my_output.jpg")
        print("This will save output to output_images/my_output.jpg on your host.")
        print("\nTo specify model path or confidence threshold via environment variables:")
        print("export MODEL_PATH=\"./path/to/your/loopvision_2025Jun25.pth\"")
        print("export CONFIDENCE_THRESHOLD=\"0.7\"")
        print("python inference.py test_image.jpg")

    # The script will exit after processing the provided image or showing usage.
    # For an API, this `if __name__ == "__main__":` block would start a web server.