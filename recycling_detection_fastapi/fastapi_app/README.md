
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





==========

# Deploy to GCP:
gcloud auth login

gcloud components update

gcloud config set project ai-platform-453201
gcloud config set compute/region asia-northeast3

### Configure Docker to authenticate with Google Cloud (if not already done)
gcloud auth configure-docker asia-northeast3-docker.pkg.dev


# (if necessary) create artifact repo
```
gcloud artifacts repositories create docker-ai-model-repo \
  --repository-format=docker \
  --location=asia-northeast3 \
  --description="Docker repository for AI models"
```


### clean memory
docker system prune --all --volumes --force


# Build and Push Your Docker Image with GPUs to Artifact Registry:

Navigate to your project directory (where `Dockerfile`, `main.py`, `requirements.txt`, `model.pth` are located).

### Define your image name for Artifact Registry
(Format: LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY_NAME/IMAGE_NAME:TAG)

<!-- 
IMAGE_NAME="asia-northeast3-docker.pkg.dev/ai-platform-453201/docker-ai-model-repo/loopvision-inference-service:v1" 
-->

IMAGE_NAME="asia-northeast3-docker.pkg.dev/ai-platform-453201/docker-ai-model-repo/loopvision-inference-api-gpu:v1"


### Build the Docker image

docker build -f Dockerfile.gpu --no-cache -t "${IMAGE_NAME}" .

### Push the image to Artifact Registry
docker push "${IMAGE_NAME}"

see:
https://console.cloud.google.com/artifacts?authuser=3&inv=1&invt=Ab2R_w&project=ai-platform-453201&supportedpurview=project


<!-- 
# Deploy to Cloud Run

  --memory 2Gi \ # Adjust based on your model size (e.g., 2Gi, 4Gi, 8Gi)
  --cpu 2 \     # Adjust based on inference demand, consider # of Gunicorn workers
  --min-instances 0 \ # Scale to zero when idle
  --max-instances 1 \ # Start with a low max instance count for testing, increase later
  --port 5050 \   # Must match the EXPOSE and Gunicorn port in your Dockerfile


gcloud run deploy loopvision-inference-service \
  --image "${IMAGE_NAME}" \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --memory 8Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 8 \
  --port 5050 \
  --set-env-vars MODEL_PATH="./loopvision_2025Jun25.pth"


Get the output URL endpoint for predictions.
Example:
Service URL: https://loopvision-inference-service-3493613107.asia-northeast3.run.app

See:
https://console.cloud.google.com/run?referrer=search&authuser=3&project=ai-platform-453201&supportedpurview=project&inv=1&invt=Ab2SAA

# Test Cloud Run API:

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  https://loopvision-inference-service-3493613107.asia-northeast3.run.app/predict 

-->




# GPU Endpoint Deployment

# Deploy to Google Kubernetes Engine (GKE) with GPU Nodes:


### Enable GKE API and Request GPU Quota:
gcloud services enable container.googleapis.com

gcloud components install gke-gcloud-auth-plugin

### Create a GKE Cluster (if you don't have one):
--num-nodes 0 \ # Start with 0 initial CPU nodes in default pool

gcloud container clusters create loopvision-gpu-cluster \
  --region asia-northeast3 \
  --release-channel "regular" \
  --num-nodes 1 \
  --node-locations asia-northeast3-b,asia-northeast3-c

### See the clusters generated:
$ gcloud container clusters list --region asia-northeast3

This command will add/update the cluster's credentials in your ~/.kube/config file.
$ gcloud container clusters get-credentials loopvision-gpu-cluster --region asia-northeast3


Then, try a basic kubectl command to list nodes:
$ kubectl get nodes



## Create a GPU Node Pool:

  --machine-type n1-standard-8 \ # Or a suitable machine type for T4s (e.g., n1-standard-X, g2-standard-X)
  --accelerator "type=nvidia-tesla-t4,count=1" \ # 1 T4 GPU per node
  --num-nodes 1 \ # Start with 1 node, can scale up
  --min-nodes 1 \ # Keep at least one GPU node warm
  --max-nodes 4 \ # Scale up to 4 GPU nodes (total 4 GPUs)
  --enable-autoscaling \
  --node-locations asia-northeast3-a,asia-northeast3-b \ # Should be in zones where T4s are available
  --disk-size 100GB \ # Ensure enough disk space for the large image

gcloud container node-pools create gpu-node-pool \
  --cluster loopvision-gpu-cluster \
  --region asia-northeast3 \
  --machine-type n1-standard-8 \
  --accelerator "type=nvidia-tesla-t4,count=1" \
  --num-nodes 1 \
  --min-nodes 1 \
  --max-nodes 4 \
  --enable-autoscaling \
  --node-locations asia-northeast3-b,asia-northeast3-c \
  --node-version 1.32.4-gke.1415000 \
  --disk-size 100GB


See:
https://console.cloud.google.com/kubernetes/list/overview?authuser=3&inv=1&invt=Ab2UJg&project=ai-platform-453201&supportedpurview=project

https://container.googleapis.com/v1/projects/ai-platform-453201/zones/asia-northeast3/clusters/loopvision-gpu-cluster/nodePools/gpu-node-pool


### Apply the Kubernetes Manifests:

kubectl apply -f k8s-deployment.yaml

kubectl get service loopvision-gpu-api-service



# Test Predictions in the endpoint:

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  http://34.64.193.133/predict





# Stop Charges:

## GKE Cluster
GKE clusters and their node pools are a major source of cost. To stop charges without deleting the cluster, you should scale the node pools down to zero nodes. This keeps the cluster's control plane running (which has a small, static cost) but stops all charges for the actual worker machines.

Scale Down Node Pools to Pause
To scale down the gpu-node-pool to 0 nodes:

$ gcloud container clusters resize loopvision-gpu-cluster --node-pool gpu-node-pool --num-nodes=0 --region asia-northeast3

To reactivate by scaling back up to a specific number of nodes (e.g., 1):

$ gcloud container clusters resize loopvision-gpu-cluster --node-pool gpu-node-pool --num-nodes=1 --region asia-northeast3

This will provision a new node, and your deployment will automatically get scheduled on it.


# Cloud Run Service
Cloud Run is a serverless platform, so it scales down to zero instances by default when it's not being used. The simplest way to "pause" it and stop all charges is to set the minimum number of instances to zero.

To check the current configuration:

$ gcloud run services describe loopvision-inference-service --region asia-northeast3
Look for the min-instances setting.

To set min-instances to 0 (pause):

gcloud run services update loopvision-inference-service --min-instances=0 --region asia-northeast3
To set min-instances back to 1 or higher (reactivate):

Bash

gcloud run services update loopvision-inference-service --min-instances=1 --region asia-northeast3
This will spin up a warm instance again, eliminating the cold start for the next request.



## Artifact Registry
Artifact Registry is a storage service. It charges based on the amount of data stored and network egress. The only way to stop these charges is to delete the stored images or the repository itself. You can't "pause" a repository.
To delete the entire repository and all images within it:

$ gcloud artifacts repositories delete docker-ai-model-repo --location=asia-northeast3 --quiet

If you do this, you'll need to rebuild and push your Docker image again when you resume.










======



# Deploy to AWS (AWS App Runner)

Prerequisites:
- AWS Account: With billing enabled.
- AWS CLI: Installed and configured on your local machine (aws configure).



# Build and Push Your Docker Image to Amazon Elastic Container Registry (ECR):

Navigate to project directory.
*Replace `YOUR_AWS_ACCOUNT_ID` and `YOUR_AWS_REGION` with your actual values.*

```
# 1. Authenticate Docker to your ECR registry (replace YOUR_AWS_ACCOUNT_ID and YOUR_AWS_REGION)
aws ecr get-login-password --region YOUR_AWS_REGION | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com

# 2. Create an ECR repository (if you haven't already)
aws ecr create-repository --repository-name loopvision-inference-api --region YOUR_AWS_REGION

# 3. Build and tag your Docker image for ECR
docker build -t loopvision-inference-api:v1 .
docker tag loopvision-inference-api:v1 YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com/loopvision-inference-api:v1

# 4. Push the image to ECR
docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_AWS_REGION.amazonaws.com/loopvision-inference-api:v1
```



# Deploy to AWS App Runner:


You can deploy via the AWS Management Console or AWS CLI. The Console is often easier for the first time.

1.  Go to the **AWS App Runner** service.
2.  Click **Create service**.
3.  **Source & Deployment:**
    * Choose **Container image**.
    * **Provider:** Amazon ECR.
    * **Browse ECR:** Select your `loopvision-inference-api` repository and the `v1` tag.
    * **Deployment trigger:** Manual (for initial testing).
4.  **Service settings:**
    * **Service name:** `loopvision-inference-service`
    * **Port:** `5000` (matches your `Dockerfile` and FastAPI app)
5.  **Compute:**
    * **CPU:** 1 vCPU (start here, monitor, scale up if needed)
    * **Memory:** 2 GB (start here, monitor, scale up if needed)
6.  **Auto scaling:**
    * **Min concurrency:** 1 (or 0 if you want it to scale down completely when idle, but be aware of cold starts)
    * **Max concurrency:** Adjust based on expected load.
7.  **Environment variables:**
    * Add `MODEL_PATH` = `./loopvision_2025Jun25.pth`
8.  **Security:**
    * Create a new **Service Role** (App Runner needs permissions to pull images from ECR).
9.  **Networking:**
    * Keep default settings (public endpoint).
10. Review and **Create & Deploy**.

Once "Running," you'll get a **Default domain** URL.

