
# Test the FastAPI locally

### setup:
conda create -n fastapi-env python=3.11

conda activate fastapi-env

export PYTORCH_ENABLE_MPS_FALLBACK=1

install dependencies:
pip install -r requirements.txt


## run locally:

conda activate fastapi-env
(conda activate /Users/oysterable/anaconda3/envs/fastapi-env)

pip install -r requirements.txt 

export PYTORCH_ENABLE_MPS_FALLBACK=1 

uvicorn main:app --host 0.0.0.0 --port 8000 --reload

### Access local server:
You can access local server at http://localhost:8000.


A. Health Check (GET /)
This is a simple request to verify that your application is live and the model has loaded successfully.

curl http://localhost:8000/

Expected Response:
{"status":"healthy","model_loaded":true}


B. Predict (POST /predict)
This request sends an image file to your /predict endpoint and expects a JSON response with the detected objects.

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/recycling_detection_fastapi/fastapi_app/test_can1.png" \
  http://localhost:8000/predict


Expected Response:

A JSON object containing an array of detections.

{
  "detections": [
    {
      "box": { "x1": 493.94, "y1": 297.65, "x2": 781.37, "y2": 590.48 },
      "confidence": 0.9589,
      "class_id": 0,
      "label": "CANS"
    }
    // ... potentially more detections
  ]
}


C. Visualize and Predict (POST /visualize_predict)
This request sends an image and expects an annotated image in return. You can save this image to a local file using curl's --output flag.


curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/recycling_detection_fastapi/fastapi_app/test_can1.png" \
  http://localhost:8000/visualize_predict \
  --output /Users/oysterable/delete/recyclables-detector/recycling_detection_fastapi/fastapi_app/annotated_prediction.png
  
--output annotated_prediction.png: This flag tells curl to save the received image data to a file named annotated_prediction.png in your current directory. You can then open this file to see the visual results.




## Deploy locally with Docker (local [cpu]):
docker build -f Dockerfile.cpu --no-cache -t loopvision-local-cpu-app .

if errors, clean up unused space:
  docker system prune --all --volumes --force




### Run the docker image:

The docker run command's port mapping must match the port specified in your main.py code. The syntax is -p [host port]:[container port].


docker run --rm \
  -v "$(pwd)/annotations:/app/annotations" \
  -p 5050:5050 \
  loopvision-local-cpu-app


### Access local server:
You can access the Docker container at http://localhost:5050.



### test predictions

[pet]

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  http://localhost:5050/predict


[can]

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/captured_1730349144841.png" \
  http://localhost:5050/predict


### Visualize Prediction

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/captured_1730349144841.png" \
  http://localhost:5050/visualize_predict --output /Users/oysterable/delete/recyclables-detector/recycling_detection_fastapi/fastapi_app/annotated_prediction.png







