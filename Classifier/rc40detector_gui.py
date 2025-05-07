# dependency install: pip install torch torchvision pillow matplotlib pyinstaller

#run
# python rc40detector_gui.py

#executable
# pyinstaller --onefile --noconsole rc40detector_gui.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tkr
from tkinter import filedialog, messagebox
import os

# ---- Configuration ----
MODEL_PATH = 'rc40classifier_2025May05.pth'
CLASS_NAMES = ['CAN', '-', 'PET']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load model from state_dict
def load_model(model_path):
    try:
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        if DEVICE.type == 'cuda':
            dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
            model(dummy)  # warm-up

        return model
    except Exception as e:
        messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
        exit()


# Preprocess image for model input
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image


# Run prediction
def predict(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]


# GUI App
class RC40App:
    def __init__(self, root):
        self.root = root
        self.root.title("RC40 Detector")
        self.root.geometry("600x600")
        self.root.resizable(False, False)

        self.model = load_model(MODEL_PATH)

        self.image_label = tkr.Label(root)
        self.image_label.pack(pady=20)

        self.prediction_label = tkr.Label(root, text="", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        self.select_button = tkr.Button(root, height=3, text="Select Image", command=self.select_image, font=("Helvetica", 14))
        self.select_button.pack(pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not file_path:
            return

        try:
            tensor, pil_image = preprocess_image(file_path)
            prediction = predict(self.model, tensor)

            # Show prediction
            self.prediction_label.config(text=f"Prediction: {prediction}")

            # Show image
            display_image = pil_image.resize((512, 512))
            tk_image = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")


# Run the app
if __name__ == '__main__':
    root = tkr.Tk()
    app = RC40App(root)
    root.mainloop()
