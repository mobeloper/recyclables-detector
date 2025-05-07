Image Classifier Package (Windows Setup)

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
