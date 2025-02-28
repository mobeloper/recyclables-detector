import os
import sys
import cv2
import csv
import logging
import zipfile
import numpy as np
import tensorflow as tf
import gradio as gr
from typing import Any, TypedDict, Callable

# Disable excessive logging
logging.disable(logging.WARNING)

# --- Clone and Setup TensorFlow Models Repository ---
if not os.path.exists("models"):
    os.system("git clone --depth 1 https://github.com/tensorflow/models 2>/dev/null")
sys.path.append('models/research/')
# Import the Object Detection API utilities
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils

# --- Utility Types and Functions ---
class ItemDict(TypedDict):
    id: int
    name: str
    supercategory: str

def load_model(model_path: str) -> Callable:
    """Loads a TensorFlow SavedModel and returns a function for making predictions."""
    try:
        print('loading model...')
        model = tf.saved_model.load(model_path)
        print('model loaded!')
        detection_fn = model.signatures['serving_default']
        return detection_fn
    except (OSError, ValueError, KeyError) as e:
        print(f"Error loading model: {e}")
        raise

def perform_detection(model: Callable, image: np.ndarray) -> dict[str, np.ndarray]:
    """Perform Mask R-CNN object detection on an image using the specified model."""
    detection_results = model(image)
    detection_results = {key: value.numpy() for key, value in detection_results.items()}
    return detection_results

def _read_csv_to_list(file_path: str) -> list[str]:
    """Reads a CSV file and returns its contents as a list."""
    data_list = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_list.append(row[0])  # Assuming there is only one column in the CSV
    return data_list

def _categories_dictionary(objects: list[str]) -> dict[int, ItemDict]:
    """Converts a list of object names into a category dictionary."""
    category_index = {}
    for num, obj_name in enumerate(objects, start=1):
        obj_dict = {'id': num, 'name': obj_name, 'supercategory': 'objects'}
        category_index[num] = obj_dict
    return category_index

def load_labels(labels_path: str) -> tuple[list[str], dict[int, ItemDict]]:
    """Load label mappings from a CSV file and generate category indices."""
    labels = _read_csv_to_list(labels_path)
    category_index = _categories_dictionary(labels)
    return labels, category_index

def filter_detection(results: dict[str, np.ndarray], valid_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Filter the detection results based on the valid indices."""
    if np.array(valid_indices).dtype == bool:
        new_num_detections = int(np.sum(valid_indices))
    else:
        new_num_detections = len(valid_indices)
    keys_to_filter = [
        'detection_masks',
        'detection_masks_resized',
        'detection_masks_reframed',
        'detection_classes',
        'detection_boxes',
        'normalized_boxes',
        'detection_scores',
        'detection_classes_names',
    ]
    filtered_output = {}
    for key in keys_to_filter:
        if key in results:
            if key == 'detection_masks':
                filtered_output[key] = results[key][:, valid_indices, :, :]
            elif key in ['detection_masks_resized', 'detection_masks_reframed']:
                filtered_output[key] = results[key][valid_indices, :, :]
            elif key in ['detection_boxes', 'normalized_boxes']:
                filtered_output[key] = results[key][:, valid_indices, :]
            elif key in ['detection_classes', 'detection_scores', 'detection_classes_names']:
                filtered_output[key] = results[key][:, valid_indices]
    filtered_output['num_detections'] = np.array([new_num_detections])
    return filtered_output

def reframe_masks(results: dict[str, np.ndarray], boxes: str, height: int, width: int) -> np.ndarray:
    """Reframe the masks to an image size."""
    detection_masks = results['detection_masks'][0]
    detection_boxes = results[boxes][0]
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, height, width
    )
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, np.uint8)
    detection_masks_reframed = detection_masks_reframed.numpy()
    return detection_masks_reframed

def _calculate_area(mask: np.ndarray) -> int:
    """Calculate the area of the mask."""
    return np.sum(mask)

def _calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate the intersection over union (IoU) between two masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def _is_contained(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Check if mask1 is entirely contained within mask2."""
    return np.array_equal(np.logical_and(mask1, mask2), mask1)

def filter_masks(masks: np.ndarray, iou_threshold=0.8, area_threshold=None) -> np.ndarray:
    """Filter the overlapping masks based on area and IoU."""
    areas = np.array([_calculate_area(mask) for mask in masks])
    sorted_indices = np.argsort(areas)[::-1]
    sorted_masks = masks[sorted_indices]
    sorted_areas = areas[sorted_indices]
    unique_indices = []
    for i, mask in enumerate(sorted_masks):
        # Optionally skip masks based on area thresholds
        if (area_threshold is not None and sorted_areas[i] > area_threshold) or sorted_areas[i] < 4000:
            continue
        keep = True
        for j in range(i):
            if _calculate_iou(mask, sorted_masks[j]) > iou_threshold or _is_contained(mask, sorted_masks[j]):
                keep = False
                break
        if keep:
            unique_indices.append(sorted_indices[i])
    return unique_indices

def adjust_image_size(height: int, width: int, min_size: int) -> tuple[int, int]:
    """Adjust the image size to ensure both dimensions are at least min_size."""
    if height < min_size or width < min_size:
        return height, width
    scale_factor = min(height / min_size, width / min_size)
    new_height = int(height / scale_factor)
    new_width = int(width / scale_factor)
    return new_height, new_width

def display_bbox_masks_labels(
    result: dict[Any, np.ndarray],
    image: np.ndarray,
    category_index: dict[int, dict[str, str]],
    threshold: float,
) -> np.ndarray:
    """Visualizes bounding boxes, masks, and labels on the image."""
    image_new = image.copy()
    # Convert from BGR to RGB for visualization
    image_new = cv2.cvtColor(image_new, cv2.COLOR_BGR2RGB)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_new,
        result['normalized_boxes'][0],
        (result['detection_classes'][0] + 0).astype(int),
        result['detection_scores'][0],
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=100,
        min_score_thresh=threshold,
        agnostic_mode=False,
        instance_masks=result.get('detection_masks_reframed', None),
        line_thickness=4,
    )
    return image_new

# --- Download and Load the Pre-trained Model ---
MODEL_DIR = "Jan2025_ver2_merged_1024_1024"
if not os.path.exists(MODEL_DIR):
    zip_path = f"{MODEL_DIR}.zip"
    if not os.path.exists(zip_path):
        os.system(f"wget https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/{MODEL_DIR}.zip -q")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
detection_fn = load_model(MODEL_DIR)

# --- Load Label Map Data ---
LABELS_PATH = (
    'models/official/projects/waste_identification_ml/pre_processing/'
    'config/data/45_labels.csv'
)
labels, category_index = load_labels(LABELS_PATH)

# --- Constants ---
HEIGHT = 1024
WIDTH = 1024
PREDICTION_THRESHOLD = 0.50

# --- Main Inference Function for Gradio ---
def process_image(image: np.ndarray) -> np.ndarray:
    """
    Accepts an input image (as a numpy array in RGB format), resizes it,
    performs inference using the Mask R-CNN model, processes detections,
    and returns an annotated image.
    """
    # Convert input from RGB to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original_image = image_bgr.copy()
    # Resize image to the dimensions the model expects
    resized_image = cv2.resize(image_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    input_tensor = tf.convert_to_tensor(np.expand_dims(resized_image, axis=0), dtype=tf.uint8)
    
    # Run detection
    result = perform_detection(detection_fn, input_tensor)
    
    if result["num_detections"][0]:
        scores = result["detection_scores"][0]
        filtered_indices = scores > PREDICTION_THRESHOLD
        result = filter_detection(result, filtered_indices)
    
    if result["num_detections"][0]:
        result["normalized_boxes"] = result["detection_boxes"].copy()
        result["normalized_boxes"][:, :, [0, 2]] /= HEIGHT
        result["normalized_boxes"][:, :, [1, 3]] /= WIDTH
        
        # Adjust image size for visualization
        height_plot, width_plot = adjust_image_size(original_image.shape[0], original_image.shape[1], 1024)
        image_plot = cv2.resize(original_image, (width_plot, height_plot), interpolation=cv2.INTER_AREA)
        
        # Reframe the masks to the new size and filter overlapping masks
        result["detection_masks_reframed"] = reframe_masks(result, "normalized_boxes", height_plot, width_plot)
        unique_indices = filter_masks(result["detection_masks_reframed"], iou_threshold=0.08, area_threshold=None)
        result = filter_detection(result, unique_indices)
    
    # Create annotated image with boxes, masks, and labels
    annotated_image = display_bbox_masks_labels(result, image_plot, category_index, PREDICTION_THRESHOLD)
    return annotated_image



css = """
footer {display: none !important;}
.gradio-container {min-height: 0 !important;}
"""

# --- Build the Gradio Blocks App ---
with gr.Blocks(css=css,analytics_enabled=False,theme=gr.themes.Ocean()) as interface:
    gr.Markdown("# ♻️ Recyclables Detector")
    gr.Markdown("Upload an image and view waste instance segmentation predictions using Mask R-CNN.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Image")
            run_button = gr.Button("Detect Objects")
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detected Objects")
    
    run_button.click(fn=process_image, inputs=input_image, outputs=output_image)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=True
    )
