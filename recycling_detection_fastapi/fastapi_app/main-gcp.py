# main.py

# ## This file includes the following features:
# - The FastAPI framework with endpoints for prediction and visualization.
# - The AI model loading logic using pretrain_weights.
# - An on_event("startup") function to load the model once.
# - torch.inference_mode() for optimal performance.
# - A data pipeline to save incoming predictions as COCO-formatted annotations  #   and save the corresponding images.
# - A local development server running on port 8000.
# - Save predictions into Google Storage Bucket for future model re-training.

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
from google.cloud import storage # NEW IMPORT

# --- Configuration ---
NUM_CLASSES = 2
MODEL_PATH = os.environ.get("MODEL_PATH", "./loopvision_2025Jun25.pth")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
CLASS_NAMES = ["CANS", "PET"]

# Use environment variables for GCS configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "loopvision-predictions-bucket")
GCS_ANNOTATIONS_BLOB = "annotated_predictions.json"
GCS_IMAGES_DIR = "images"

CATEGORIES = [{"id": 0, "name": "CANS"}, {"id": 1, "name": "PET"}]

app = FastAPI(
    title="RFDETRBase Inference API",
    description="API for detecting CANS and PET using a custom RFDETRBase model.",
    version="1.0.0",
)

model = None
coco_annotation_id_counter = 0

# Initialize GCS client
storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

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
            
            # --- Check for existing annotations in GCS ---
            annotations_blob = bucket.blob(GCS_ANNOTATIONS_BLOB)
            if annotations_blob.exists():
                with annotations_blob.open("r") as f:
                    data = json.load(f)
                    if 'annotations' in data and data['annotations']:
                        coco_annotation_id_counter = max([ann['id'] for ann in data['annotations']]) + 1
                    else:
                        coco_annotation_id_counter = 1
            else:
                coco_annotation_id_counter = 1

        except Exception as e:
            print(f"Error loading model or GCS annotations: {e}")
            sys.exit(1)

# --- Function to save the image to GCS ---
def save_image_to_gcs(image_bytes: bytes, image_filename: str):
    """Saves the image bytes to GCS."""
    blob_path = os.path.join(GCS_IMAGES_DIR, image_filename)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(image_bytes, content_type='image/png')
    return blob_path

# --- Function to save predictions to GCS in COCO format ---
def save_as_coco_annotation(
    detections: sv.Detections, 
    image: Image.Image, 
    image_filename: str
):
    global coco_annotation_id_counter
    annotations_blob = bucket.blob(GCS_ANNOTATIONS_BLOB)
    
    if annotations_blob.exists():
        with annotations_blob.open("r") as f:
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
    
    # Store the file path in the JSON to follow the desired structure in GCS
    data['images'].append({
        "id": image_id,
        "file_name": os.path.join(GCS_IMAGES_DIR, image_filename),
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

    # Save the updated data back to the GCS blob
    with annotations_blob.open("w") as f:
        json.dump(data, f, indent=4)

# --- /predict endpoint ---
@app.post("/predict", response_model=Dict[str, List[Dict[str, Any]]])
async def predict_image_api(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image type (e.g., image/jpeg, image/png)")

    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with torch.inference_mode():
            detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)

        # --- Save image and annotations to GCS ---
        save_image_to_gcs(image_bytes, image.filename)
        save_as_coco_annotation(detections, pil_image, image.filename)
        print(f"Annotations and image saved for file: {image.filename} to GCS bucket: {GCS_BUCKET_NAME}")
        
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
    
    