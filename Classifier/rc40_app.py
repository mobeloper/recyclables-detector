# install requirements:
#pip install -r requirements.txt

# run:
# streamlit run rc40_app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import os
import time

# ---- Page Configuration ----
st.set_page_config(page_title="RC40 Recyclables Classifier", layout="wide")

# ---- Custom CSS Styling ----
STYLES = """
<style>
/* Global background */
[data-testid="stAppViewContainer"] { 
/* background-color: #f7f9fc; */
background-color: black;
}
/* Sidebar styling */
[data-testid="stSidebar"] { 
background-color: #ffffff; 
}
/* Title styling */
h1[data-testid="stTitle"] { 
font-family: 'Helvetica Neue', sans-serif; color: #ababab; 
}
/* Radio buttons container */
div[role="radiogroup"] { 
background-color: #3EA499; 
border-radius: 8px; 
padding: 10px; 
}
/* Buttons styling */
.stButton>button {
    background-color: #3EA499;
    color: black;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease;
}
.stButton>button:hover { background-color: white; }
/* Image styling */
.stImage img {
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
/* Prediction text overlay */
.pred-overlay {
    position: absolute;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    background-color: rgba(0, 0, 0, 0.6);
    color: white;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 18px;
    font-family: 'Helvetica Neue', sans-serif;
}

/* Prediction text styling */
.pred-center {
    text-align: center;
    font-size: 24px;
    font-width: bold;
    font-family: 'Helvetica Neue', sans-serif;
    color: white;
    margin-bottom: 10px;
}

</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ---- Configuration ----
MODEL_PATH = 'rc40classifier_2025May05.pth'
CLASS_NAMES = ['CAN', '-', 'PET']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache the model resource using new decorator
@st.cache_resource
def load_model(path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_pil):
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# Initialize session state
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'images' not in st.session_state:
    st.session_state.images = []
if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'folder' not in st.session_state:
    st.session_state.folder = ''

# App UI
st.title("RC40 Recyclables Classifier")
# Wide two-column layout covering full width
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    mode = st.radio("Input Method", ("Upload Image", "Select Folder", "Use Camera"))

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

    elif mode == "Select Folder":
        folder_path = st.text_input("Enter full folder path: (ex.  /Users/oysterable/images/)")
        if folder_path and os.path.isdir(folder_path):
            # Reload if folder changed
            if st.session_state.folder != folder_path:
                st.session_state.folder = folder_path
                st.session_state.images = sorted([
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                st.session_state.idx = 0

            prev_col, next_col = st.columns(2)
            if prev_col.button("Previous") and st.session_state.idx > 0:
                st.session_state.idx -= 1
            if next_col.button("Next") and st.session_state.idx < len(st.session_state.images) - 1:
                st.session_state.idx += 1

    else:  # Use Camera
        start, stop = st.columns(2)
        if start.button("Start Camera"):
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)
        if stop.button("Stop Camera"):
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None


with col2:
    pred_placeholder = st.empty()
    img_placeholder = st.empty()

    if mode == "Upload Image" and 'uploaded_file' in locals() and uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        pred = predict_image(img)
        pred_placeholder.markdown(f"<div class='pred-center'>Prediction: {pred}</div>", unsafe_allow_html=True)
        img_placeholder.image(img, use_column_width=True)
        st.markdown(f"**Filename:** {uploaded_file.name}")

    elif mode == "Select Folder" and st.session_state.images:
        current_image_path = st.session_state.images[st.session_state.idx]
        img = Image.open(current_image_path).convert('RGB')
        pred = predict_image(img)
        pred_placeholder.markdown(f"<div class='pred-center'>Prediction: {pred}</div>", unsafe_allow_html=True)
        img_placeholder.image(img, use_column_width=True)
        st.markdown(f"**Filename:** {os.path.basename(current_image_path)}")

    elif mode == "Use Camera" and st.session_state.cap:
        cap = st.session_state.cap
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            pred = predict_image(pil_img)
            pred_placeholder.markdown(f"<div class='pred-center'>Prediction: {pred}</div>", unsafe_allow_html=True)
            img_placeholder.image(img, use_column_width=True)
            time.sleep(0.03)
            if st.session_state.cap is None:
                break

    else:
        pred_placeholder.markdown("<div class='pred-center'>No input selected.</div>", unsafe_allow_html=True)

# Cleanup on exit
if st.session_state.cap is None and 'cap' in locals() and locals()['cap']:
    locals()['cap'].release()