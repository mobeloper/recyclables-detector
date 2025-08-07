
# main.py
import sys
import os
import torch
from PIL import Image
import supervision as sv
from rfdetr import RFDETRBase
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io # To handle image bytes
from typing import List, Dict, Any

# --- Configuration (can be moved to environment variables for production) ---
NUM_CLASSES = 2
# MODEL_PATH will be expected relative to the container's /app directory
MODEL_PATH = os.environ.get("MODEL_PATH", "./loopvision_2025Jun25.pth")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.5))
CLASS_NAMES = ["CANS", "PET"]

# Initialize FastAPI app
app = FastAPI(
    title="RFDETRBase Inference API",
    description="API for detecting CANS and PET using DETR architecture model.",
    version="1.0.1",
)

# Global variable to hold the loaded model
model = None

@app.on_event("startup")
async def startup_event():
    """
    Loads the RFDETRBase model when the FastAPI application starts up.
    """
    global model
    if model is None:
        print(f"Attempting to load RFDETRBase model with {NUM_CLASSES} classes from: {MODEL_PATH}")
        try:
            # Initialize the model architecture with the correct number of classes
            # RFDETRBase loads the .pth weights internally via pretrain_weights
            # As per your preferred loading method and specific library behavior
            model = RFDETRBase(
                num_classes=NUM_CLASSES,
                pretrain_weights=MODEL_PATH
            )

            # --- REMOVED: model.to(device) and model.eval() as per user's instruction ---
            # RFDETRBase handles moving the model to GPU if CUDA is available,
            # and its internal methods are expected to manage evaluation mode.

            print("RFDETRBase model loaded successfully with custom weights.")

        except Exception as e:
            print(f"Error loading RFDETRBase model: {e}")
            sys.exit(1) # Critical to exit if model fails to load


@app.post("/predict", response_model=Dict[str, List[Dict[str, Any]]])
async def predict_image_api(image: UploadFile = File(...)):
    """
    Performs object detection on an uploaded image.

    - **image**: The image file to process.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image type (e.g., image/jpeg, image/png)")

    try:
        # Read image data from the UploadFile
        image_bytes = await image.read() # Use await for async file read
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Perform inference using the loaded model
        # detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)
        with torch.inference_mode():
            detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)


        # Prepare results for JSON response
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

@app.get("/")
async def health_check():
    """
    Health check endpoint to ensure the API is running.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy", "model_loaded": True}

# Note: The `if __name__ == '__main__':` block is typically for local testing
# and debugging. For production, Uvicorn will be started by the Docker CMD.
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
