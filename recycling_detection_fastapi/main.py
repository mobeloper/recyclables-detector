import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import cv2
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
import supervision as sv # Still needed for generic SV utilities if used for drawing
import asyncio
import io
import base64

# Assuming rfdetr is installed or available in your environment
from rfdetr import RFDETRBase

# --- Configuration ---
NUM_CLASSES = 2
CLASSES = ["CANS", "PET"]
CONFIDENCE_THRESHOLD = 0.6

# Class specific colors (B, G, R) for OpenCV
CLASS_COLORS_BGR = {
    "CANS": (0, 0, 255),   # Red for CANS
    "PET": (255, 0, 0)     # Blue for PET
}

# Column definitions (normalized coordinates 0-1)
COLUMN_WIDTH = 1/2
COLUMNS = {
    "CANS": {"x_min": 0, "x_max": COLUMN_WIDTH},
    "PET": {"x_min": COLUMN_WIDTH, "x_max": 1}
}

# Reference line position (from bottom of the frame)
REFERENCE_LINE_HEIGHT_RATIO = 0.3

app = FastAPI()

# --- Model Loading ---
MODEL_PATH = "./model/loopvision_2025Jun25.pth"
model = None
try:
    print(f"[DEBUG] Attempting to load model from: {MODEL_PATH}")
    model = RFDETRBase(
        num_classes=NUM_CLASSES,
        pretrain_weights=MODEL_PATH
    )
    print(f"Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'rfdetr' is correctly installed and 'loopvision_2025Jun25.pth' exists.")
    exit()


# --- Global Counters ---
object_counts = {
    "Total": 0,
    "PET": 0,
    "CANS": 0
}

last_detected_object_bbox = None
object_id_counter = 0
active_objects = {}


# --- HTML for the Frontend ---
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Lalaloop AI Recycling</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1A1A2E; /* Dark Blue */
            color: #E0E0E0; /* Light Gray */
            margin: 0;
            overflow: hidden; /* Hide scrollbars if content overflows */
            display: flex; /* Use flexbox to center content */
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh; /* Ensure body takes full viewport height */
            width: 100vw; /* Ensure body takes full viewport width */
        }
        .container {
            position: relative; /* For absolute positioning of video and overlays */
            width: 100vw; /* Container takes full viewport width */
            height: 100vh; /* Container takes full viewport height */
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background-color: #2D2D44; /* Darker Blue/Gray */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        #videoFeed {
            position: absolute; /* Position absolutely to cover container */
            top: 0;
            left: 0;
            width: 100%; /* Make video fill its parent container */
            height: 100%; /* Make video fill its parent container */
            object-fit: cover; /* This is key: crops to cover, maintains aspect ratio */
            border: none; /* Remove border to maximize screen space */
            border-radius: 0; /* Remove border-radius */
            z-index: 0; /* Place behind other UI elements */
        }
        .header-title {
            position: absolute;
            top: 20px; /* Adjust as needed */
            font-size: 2.5em;
            color: #00C3B3;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
            z-index: 10; /* Place on top of video */
        }
        .counts-container {
            position: absolute;
            bottom: 20px; /* Adjust as needed */
            display: flex;
            justify-content: space-around;
            width: 80%; /* Adjust width to fit */
            padding: 10px 20px;
            background-color: rgba(45, 45, 68, 0.8); /* Semi-transparent background */
            border-radius: 8px;
            font-size: 1.2em;
            color: #00C3B3; /* Main UI color */
            flex-wrap: wrap;
            z-index: 10; /* Place on top of video */
        }
        .count-item {
            margin: 5px 15px;
            text-align: center;
        }
        .points-overlay {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 4em;
            font-weight: bold;
            color: #00C3B3;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
            z-index: 1000;
            display: none;
            animation: fadeOut 3s forwards;
            pointer-events: none; /* Allow clicks through */
        }
        .confetti-overlay {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 8em;
            z-index: 999;
            display: none;
            animation: fadeOut 3s forwards;
            pointer-events: none; /* Allow clicks through */
        }

        @keyframes fadeOut {
            0% { opacity: 1; }
            80% { opacity: 1; }
            100% { opacity: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header-title">Lalaloop Recycling Assistant</h1>
        <img id="videoFeed" src="" alt="Video Feed">
        <div class="counts-container">
            <div class="count-item">Total: <span id="totalCount">0</span></div>
            <div class="count-item">PET: <span id="petCount">0</span></div>
            <div class="count-item">CANS: <span id="cansCount">0</span></div>
        </div>
        <div id="pointsOverlay" class="points-overlay">+5 points</div>
        <div id="confettiOverlay" class="confetti-overlay">🎉</div>
    </div>

    <script>
        var ws = new WebSocket("ws://localhost:8000/ws");
        var videoFeed = document.getElementById('videoFeed');
        var totalCountSpan = document.getElementById('totalCount');
        var petCountSpan = document.getElementById('petCount');
        var cansCountSpan = document.getElementById('cansCount');
        var pointsOverlay = document.getElementById('pointsOverlay');
        var confettiOverlay = document.getElementById('confettiOverlay');

        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);
            if (data.image) {
                videoFeed.src = 'data:image/jpeg;base64,' + data.image;
            }
            if (data.counts) {
                totalCountSpan.textContent = data.counts.Total;
                petCountSpan.textContent = data.counts.PET;
                cansCountSpan.textContent = data.counts.CANS;
            }
            if (data.showConfetti) {
                pointsOverlay.style.display = 'block';
                confettiOverlay.style.display = 'block';
                // Reset animation
                pointsOverlay.style.animation = 'none';
                confettiOverlay.style.animation = 'none';
                void pointsOverlay.offsetWidth; // Trigger reflow
                void confettiOverlay.offsetWidth; // Trigger reflow
                pointsOverlay.style.animation = 'fadeOut 3s forwards';
                confettiOverlay.style.animation = 'fadeOut 3s forwards';

                setTimeout(() => {
                    pointsOverlay.style.display = 'none';
                    confettiOverlay.style.display = 'none';
                }, 3000);
            }
        };

        ws.onclose = function(event) {
            console.log("WebSocket closed with code: " + event.code + ", reason: " + event.reason);
        };

        ws.onerror = function(error) {
            console.error("WebSocket error: " + error);
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

async def generate_frames(websocket: WebSocket):
    global object_counts, last_detected_object_bbox, object_id_counter, active_objects

    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        try:
            await websocket.send_json({"error": "Webcam not found or accessible. Please check permissions and connectivity."})
            await websocket.close()
        except Exception as e:
            print(f"Error sending webcam error to client: {e}")
        return
    
    print(f"[DEBUG] Webcam opened successfully.")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[DEBUG] Frame resolution: {frame_width}x{frame_height}")
    reference_line_y = int(frame_height * (1 - REFERENCE_LINE_HEIGHT_RATIO))
    print(f"[DEBUG] Reference line Y position: {reference_line_y}")

    arrow_blink_state = False
    last_blink_time = asyncio.get_event_loop().time()
    blink_interval = 0.5 # seconds


    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] Failed to read frame from webcam. Exiting loop.")
            break

        # Flip frame horizontally for typical webcam view
        frame = cv2.flip(frame, 1)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        show_arrow = False
        arrow_direction = None
        arrow_column = None
        show_confetti_and_points = False

        try:
            print(f"[DEBUG] Calling model.predict() with threshold={CONFIDENCE_THRESHOLD}")
            detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)
            
            print(f"[DEBUG] Model prediction returned: {detections}")
            if detections.xyxy is not None:
                print(f"[DEBUG] Number of raw detections: {len(detections.xyxy)}")
            else:
                print("[DEBUG] No raw detections found (detections.xyxy is None).")

            detected_objects_in_frame_heuristic = {} # To hold current frame's detected objects for simple tracking


            if detections.xyxy is not None and len(detections.xyxy) > 0:
                for i, (bbox, class_id, confidence) in enumerate(zip(detections.xyxy, detections.class_id, detections.confidence)):
                    x_min, y_min, x_max, y_max = map(int, bbox)
                    class_name = CLASSES[class_id]
                    color_bgr = CLASS_COLORS_BGR.get(class_name, (255, 255, 255))

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color_bgr, 2)

                    label = f"{class_name} {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_y = y_min - 10 if y_min - 10 > text_height else y_min + text_height + 5
                    cv2.putText(frame, label, (x_min, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2, cv2.LINE_AA)

                    show_arrow = True

                    object_center_x_norm = (x_min + x_max) / 2 / frame_width
                    if object_center_x_norm < COLUMNS["CANS"]["x_max"]:
                        arrow_direction = "CANS"
                        arrow_column = (frame_width / 4)
                    else:
                        arrow_direction = "PET"
                        arrow_column = (frame_width * 3 / 4)

                    current_object_center_x = (x_min + x_max) // 2
                    current_object_bottom_y = y_max

                    # This assignment MUST happen for every detected object,
                    # regardless of whether it's above or below the line.
                    detected_objects_in_frame_heuristic[f"{class_name}_{i}"] = {
                        "bbox": bbox,
                        "class_name": class_name,
                        "center_x": current_object_center_x,
                        "bottom_y": current_object_bottom_y
                    }

                    # Check if object crosses the line
                    if current_object_bottom_y > reference_line_y:
                        print(f"[DEBUG] Object {class_name} (Y:{current_object_bottom_y}) is below reference line.")
                        is_new_crossing = True
                        for obj_key, obj_data in list(active_objects.items()):
                            last_x_center = (obj_data["last_bbox"][0] + obj_data["last_bbox"][2]) / 2
                            last_y_bottom = obj_data["last_bbox"][3]

                            if abs(current_object_center_x - last_x_center) < 50 and \
                               abs(current_object_bottom_y - last_y_bottom) < 50 and \
                               obj_data["class_name"] == class_name:
                                if not obj_data["crossed_line"]:
                                    print(f"[DEBUG] Object {class_name} (Key:{obj_key}) newly crossed line.")
                                    object_counts["Total"] += 1
                                    object_counts[class_name] += 1
                                    obj_data["crossed_line"] = True
                                    
                                    object_center_x_for_crossing = (x_min + x_max) / 2
                                    if (class_name == "CANS" and object_center_x_for_crossing < frame_width / 2) or \
                                       (class_name == "PET" and object_center_x_for_crossing >= frame_width / 2):
                                        show_confetti_and_points = True
                                        print(f"[DEBUG] +5 Points awarded for {class_name}!")
                                        
                                    print(f"Object '{class_name}' crossed the line! Total: {object_counts['Total']}")
                                is_new_crossing = False
                                break

                        if is_new_crossing:
                            print(f"[DEBUG] New object {class_name} detected and crossed line.")
                            object_id_counter += 1
                            active_objects[object_id_counter] = {
                                "last_bbox": bbox,
                                "crossed_line": True,
                                "class_name": class_name
                            }
                            object_counts["Total"] += 1
                            object_counts[class_name] += 1

                            object_center_x_for_crossing = (x_min + x_max) / 2
                            if (class_name == "CANS" and object_center_x_for_crossing < frame_width / 2) or \
                               (class_name == "PET" and object_center_x_for_crossing >= frame_width / 2):
                                show_confetti_and_points = True
                                print(f"[DEBUG] +5 Points awarded for new {class_name}!")

                            print(f"New object '{class_name}' crossed the line! Total: {object_counts['Total']}")

                    # Corrected indentation of else block to align with the 'if current_object_bottom_y > reference_line_y:'
                    else: # Object is above the line
                        # print(f"[DEBUG] Object {class_name} (Y:{current_object_bottom_y}) is above reference line.")
                        for obj_key, obj_data in list(active_objects.items()):
                            if obj_data["class_name"] == class_name and obj_key in active_objects:
                                current_obj_center = (bbox[0] + bbox[2]) / 2
                                last_obj_center = (obj_data["last_bbox"][0] + obj_data["last_bbox"][2]) / 2
                                if abs(current_obj_center - last_obj_center) < 50:
                                    if active_objects[obj_key]["crossed_line"] and current_object_bottom_y < reference_line_y - 20:
                                        print(f"[DEBUG] Object {class_name} (Key:{obj_key}) moved back above line. Resetting 'crossed_line' status.")
                                        active_objects[obj_key]["crossed_line"] = False
                                    active_objects[obj_key]["last_bbox"] = bbox
                                break


            # Update active_objects: remove objects that are no longer detected or have moved far away
            objects_to_remove_keys = []
            for obj_id, obj_data in list(active_objects.items()):
                found_in_current_frame = False
                for current_obj_key, current_obj_data in detected_objects_in_frame_heuristic.items():
                    if abs((obj_data["last_bbox"][0] + obj_data["last_bbox"][2])/2 - current_obj_data["center_x"]) < 50 and \
                       abs((obj_data["last_bbox"][1] + obj_data["last_bbox"][3])/2 - current_obj_data["bottom_y"]) < 50 and \
                       obj_data["class_name"] == current_obj_data["class_name"]:
                        found_in_current_frame = True
                        break
                if not found_in_current_frame:
                    print(f"[DEBUG] Object {obj_id} (Class: {obj_data['class_name']}) no longer detected, removing from active_objects.")
                    objects_to_remove_keys.append(obj_id)

            for obj_id in objects_to_remove_keys:
                del active_objects[obj_id]


        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()


        # Update blinking state (moved outside try-except)
        current_time = asyncio.get_event_loop().time()
        if current_time - last_blink_time > blink_interval:
            arrow_blink_state = not arrow_blink_state
            last_blink_time = current_time

        # Draw reference line
        cv2.line(frame, (0, reference_line_y), (frame_width, reference_line_y), (255, 255, 255), 2)
        # print(f"[DEBUG] Drawn reference line at Y={reference_line_y}")

        # Draw blinking arrow if applicable
        if show_arrow and arrow_blink_state and arrow_direction:
            arrow_thickness = 10
            arrow_length = 80
            arrow_head_size = 30
            arrow_y_pos = frame_height - 50
            # print(f"[DEBUG] Drawing arrow: Direction={arrow_direction}, Column_X={arrow_column}")

            if arrow_direction == "CANS":
                start_point = (int(arrow_column), arrow_y_pos)
                end_point = (int(arrow_column) - arrow_length, arrow_y_pos + arrow_length)
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), arrow_thickness, tipLength=0.5)
                cv2.line(frame, (end_point[0] + int(np.cos(np.pi/4) * arrow_head_size), end_point[1] - int(np.sin(np.pi/4) * arrow_head_size)), end_point, (0, 255, 0), arrow_thickness)
                cv2.line(frame, (end_point[0] + int(np.cos(np.pi/4) * arrow_head_size), end_point[1] + int(np.sin(np.pi/4) * arrow_head_size)), end_point, (0, 255, 0), arrow_thickness)


            elif arrow_direction == "PET":
                start_point = (int(arrow_column), arrow_y_pos)
                end_point = (int(arrow_column) + arrow_length, arrow_y_pos + arrow_length)
                cv2.arrowedLine(frame, start_point, end_point, (0, 255, 0), arrow_thickness, tipLength=0.5)
                cv2.line(frame, (end_point[0] - int(np.cos(np.pi/4) * arrow_head_size), end_point[1] - int(np.sin(np.pi/4) * arrow_head_size)), end_point, (0, 255, 0), arrow_thickness)
                cv2.line(frame, (end_point[0] - int(np.cos(np.pi/4) * arrow_head_size), end_point[1] + int(np.sin(np.pi/4) * arrow_head_size)), end_point, (0, 255, 0), arrow_thickness)


        _, buffer = cv2.imencode('.jpeg', frame)
        frame_bytes = buffer.tobytes()

        try:
            await websocket.send_json({
                "image": base64.b64encode(frame_bytes).decode('utf-8'),
                "counts": object_counts,
                "showConfetti": show_confetti_and_points
            })
            show_confetti_and_points = False
        except WebSocketDisconnect:
            print("[DEBUG] Client disconnected during frame send.")
            break
        except Exception as e:
            print(f"Error sending frame: {e}")
            import traceback
            traceback.print_exc()
            break

    cap.release()
    print("[DEBUG] Webcam released.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[DEBUG] WebSocket connection accepted.")
    try:
        await generate_frames(websocket)
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket disconnected (from endpoint handler).")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] WebSocket connection closed.")

if __name__ == "__main__":
    import base64
    print("[DEBUG] Starting Uvicorn server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)