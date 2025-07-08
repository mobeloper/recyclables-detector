

# Build docker image
docker build -t loopvision-v1-app .

# Run Docker container
docker run --rm -v $(pwd)/test_image.jpg:/app/test_image.jpg loopvision-v1-app python inference.py test_image.jpg

-v $(pwd)/test_image.jpg:/app/test_image.jpg - This mounts your local test_image.jpg into the container at /app/test_image.jpg. This is better than COPYing every test image.

>>>loopvision-v1-app python inference.py test_image.jpg - Overrides the default CMD to run your script with the mounted image.


>>For GPU: 
docker run --rm --gpus all --ipc=host loopvision-v1-app
docker run --rm --gpus all --ipc=host -v $(pwd)/test_image.jpg:/app/test_image.jpg loopvision-v1-app python inference.py test_image.jpg

or When running FastAPI app: 
docker run -p 5000:5000 --gpus all --rm loopvision-v1-app
(maps container port 5000 to host port 5000).



### Running with environment variables (e.g., custom confidence threshold):

docker run --rm --gpus all --ipc=host -v $(pwd)/test_image.jpg:/app/test_image.jpg -e CONFIDENCE_THRESHOLD=0.7 loopvision-inference python inference.py test_image.jpg



### If your model file or input data is large or frequently updated, you might not want to copy it into the image. Instead, you can mount a host directory as a volume:
docker run -v /path/to/local/models:/app/models --gpus all --rm loopvision-v1-app

>>Then, in your inference.py, you'd load the model from /app/models/my_model.pth


### You can pass configuration parameters (e.g., model path, device to use) as environment variables to your container:
docker run -e MODEL_PATH=/app/my_model.pth --rm my-pytorch-inference-app

>>> And access them in Python using os.environ.get('MODEL_PATH').