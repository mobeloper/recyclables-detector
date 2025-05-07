
#install dependencies:
# pip install torch torchvision pillow matplotlib

#run code:
# python rc40detector.py '/Users/oysterable/delete/recyclables-detector/Classifier/test-images/img-1.png'



## Optional: Build EXE
#
# pip install pyinstaller
# pyinstaller --onefile rc40detector.py
# Optional: Add --noconsole if you don't want a terminal window to appear (for GUI apps).

# Find `rc40detector.exe` in the `dist` folder.
# rc40detector.py

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import logging

# ---- Configuration ----
MODEL_PATH = 'rc40classifier_2025May05.pth'
CLASS_NAMES = ['CAN', '-', 'PET']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging setup
logging.basicConfig(filename='rc40detector.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Load model with architecture definition
def load_model(model_path):
    try:
        # model = models.resnet18(pretrained=True)
        model = models.resnet50(pretrained=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        if DEVICE.type == 'cuda':
            dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
            model(dummy)  # warm-up

        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        print(f"Error loading model: {e}")
        sys.exit(1)


# Load and transform a single image
def load_image(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return tensor, image
    except Exception as e:
        logging.error(f"Image loading failed: {e}")
        print(f"Error loading image: {e}")
        sys.exit(1)


# Run prediction
def predict(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    class_name = CLASS_NAMES[predicted.item()]
    logging.info(f"Predicted class: {class_name}")
    return class_name


# Display image with prediction
def show_image_with_prediction(image, prediction):
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()


# Optional batch support (folder of images)
def batch_inference(model, folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            tensor, image = load_image(path)
            prediction = predict(model, tensor)
            print(f"{filename} => {prediction}")
            show_image_with_prediction(image, prediction)


# Entry point
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rc40detector.py <image_path_or_folder>")
        sys.exit(0)

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    if os.path.isdir(input_path):
        print("Running batch inference on folder...")
        batch_inference(model, input_path)
    else:
        print("Running prediction on single image...")
        tensor, image = load_image(input_path)
        prediction = predict(model, tensor)
        print(f"Predicted Class: {prediction}")
        show_image_with_prediction(image, prediction)
