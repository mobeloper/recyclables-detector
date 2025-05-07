import os
import shutil
from pathlib import Path

# Set up directory structure

base_dir = Path("/")
script_dir = base_dir / "ai-model"
pictures_dir = Path.home() / "Desktop" / "Pictures"

# Create necessary directories
os.makedirs(script_dir, exist_ok=True)


predict_script = """

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
CLASS_NAMES = ['CANS', 'CARDBOARD', 'PET']
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


"""


# Add README instructions
readme_content = """

# Image Classifier Package (Windows Setup)

### Read detailed instructions here: 
https://www.notion.so/oysterable/RC40-Detector-Instruction-Setup-1eb9586d4bcb808aa9ded39a6649150a


## 1. Install Python
Download and install Python from: https://www.python.org/downloads/windows/
- Check "Add Python to PATH" during installation.

## 2. Open Command Prompt
- Press `Windows + R`, type `cmd`, and press Enter.

## 3. Navigate to the Folder
Change to the directory to where the script is:
cd C:\ai-model

## 4. Create Virtual Environment and Install Requirements
python -m venv myenv
myenv\Scripts\activate
pip install torch torchvision pillow matplotlib


## 5. Place your model and images
- Put `rc40classifier_<date>.pth` and "rc40detector.py" in the `C:\ai-model` folder.
- Put your images in `C:\Users\<YourName>\Desktop\Pictures`

## 6. Run the Prediction
python predict.py "path-to-image.jpg"
ex.) python predict.py "C:\Users\Oysterable\Desktop\Pictures\img-123.jpg"

"""


# Write predict.py to folder
with open(script_dir / "rc40detector.py", "w") as f:
    f.write(predict_script)

with open(script_dir / "README.txt", "w") as f:
    f.write(readme_content)

# Create a zip file of the package
zip_path = f"{base_dir}.zip"
shutil.make_archive(str(base_dir), 'zip', root_dir=base_dir)
