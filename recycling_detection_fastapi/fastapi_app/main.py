# main.py
import sys
import os
import torch
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import io
import json
import uuid
from typing import List, Dict, Any, Union, Optional

# --- Configuration ---
NUM_CLASSES = 2
MODEL_PATH = os.environ.get("MODEL_PATH", "./loopvision_2025Jun25.pth")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
CLASS_NAMES = ["CANS", "PET"]

# Output paths are now relative to the app's root
COCO_FILE_PATH = os.path.join("annotations", "annotated_predictions.json")
IMAGE_SAVE_PATH = os.path.join("annotations", "images")

CATEGORIES = [{"id": 0, "name": "CANS"}, {"id": 1, "name": "PET"}]

app = FastAPI(
    title="RFDETRBase Inference API",
    description="API for detecting CANS and PET using a custom RFDETRBase model.",
    version="1.0.0",
)

model = None
coco_annotation_id_counter = 0

@app.on_event("startup")
async def startup_event():
    global model
    global coco_annotation_id_counter
    if model is None:

        try:
            print(f"Attempting to load RFDETRBase model with {NUM_CLASSES} classes from: {MODEL_PATH}")
            model = RFDETRBase(
                num_classes=NUM_CLASSES,
                pretrain_weights=MODEL_PATH
            )
            print("RFDETRBase model loaded successfully with custom weights.")
                        
            
            if os.path.exists(COCO_FILE_PATH):
                with open(COCO_FILE_PATH, 'r') as f:
                    data = json.load(f)
                    if 'annotations' in data and data['annotations']:
                        coco_annotation_id_counter = max([ann['id'] for ann in data['annotations']]) + 1
                    else:
                        coco_annotation_id_counter = 1
            else:
                coco_annotation_id_counter = 1

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure 'rfdetr' is correctly installed and 'loopvision_2025Jun25.pth' exists.")
            sys.exit(1)

# --- Function to save the image to disk ---
def save_image_to_disk(image_bytes: bytes, image_filename: str):
    """Saves the image bytes to a file in the designated image save path."""
    # Ensure the directory exists
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
    image_path = os.path.join(IMAGE_SAVE_PATH, image_filename)
    with open(image_path, 'wb') as f:
        f.write(image_bytes)
    return image_path

# --- Function to save predictions to COCO format ---
def save_as_coco_annotation(
    detections: sv.Detections, 
    image: Image.Image, 
    image_filename: str
):
    global coco_annotation_id_counter
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(COCO_FILE_PATH), exist_ok=True)
    
    if os.path.exists(COCO_FILE_PATH):
        with open(COCO_FILE_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {
            "info": {},
            "licenses": [],
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

    image_id = str(uuid.uuid4())
    
    data['images'].append({
        "id": image_id,
        "file_name": os.path.join(IMAGE_SAVE_PATH, image_filename), # Updated file path in JSON
        "width": image.width,
        "height": image.height
    })

    for i in range(len(detections.xyxy)):
        x1, y1, x2, y2 = detections.xyxy[i].tolist()
        confidence = detections.confidence[i].item()
        class_id = detections.class_id[i].item()
        width = x2 - x1
        height = y2 - y1
        
        data['annotations'].append({
            "id": coco_annotation_id_counter,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [x1, y1, width, height],
            "score": confidence,
            "iscrowd": 0,
            "area": width * height
        })
        coco_annotation_id_counter += 1

    with open(COCO_FILE_PATH, 'w') as f:
        json.dump(data, f, indent=4)

# --- Function to visualize detections ---
def visualize_detections(image: Image.Image, detections: sv.Detections):
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    bbox_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK, text_scale=text_scale,
        text_thickness=thickness, smart_position=True
    )
    detections_labels = [
        f"{CLASS_NAMES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    annotated_image = image.copy()
    annotated_image = bbox_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, detections_labels)
    return annotated_image


# --- /predict endpoint ---
@app.post("/predict", response_model=Dict[str, List[Dict[str, Any]]])
async def predict_image_api(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image type (e.g., image/jpeg, image/png)")

    try:
        print(f"Calling model.predict() with threshold={CONFIDENCE_THRESHOLD}")
        
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.inference_mode():
            detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)

        # --- Save image and annotations ---
        image_bytes_stream = io.BytesIO(image_bytes)
        image_bytes_stream.seek(0)
        
        save_image_to_disk(image_bytes_stream.getvalue(), image.filename)
        save_as_coco_annotation(detections, pil_image, image.filename)
        print(f"Annotations saved for file: {image.filename}")
        
        results = []
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i].tolist()
            confidence = detections.confidence[i].item()
            class_id = detections.class_id[i].item()
            label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else "unknown"

            results.append({
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": confidence,
                "class_id": class_id,
                "label": label
            })

        return JSONResponse(content={"detections": results})

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- /visualize_predict endpoint ---
@app.post("/visualize_predict")
async def visualize_predict_api(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image type (e.g., image/jpeg, image/png)")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.inference_mode():
            detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)

        annotated_image = visualize_detections(pil_image, detections)

        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        print(f"Error during visualization prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# --- Health check endpoint ---
@app.get("/")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy", "model_loaded": True}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)