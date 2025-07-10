
# Test the FastAPI locally

### setup:
conda create -n fastapi-env python=3.11

conda activate fastapi-env

export PYTORCH_ENABLE_MPS_FALLBACK=1

install dependencies:
pip install -r requirements.txt


### run locally:

conda activate fastapi-env

uvicorn main:app --host 0.0.0.0 --port 5050 --reload


### with docker:

docker build -f Dockerfile.cpu --no-cache -t loopvision-local-cpu-app .

docker run --rm \
  -v "$(pwd)/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png:/app/test_image.png" \
  -v "$(pwd)/OutputPredictions:/app/output" \
  -p 5050:5050 \
  loopvision-local-cpu-app


### test predictions
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  http://localhost:5050/predict





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
gcloud container clusters list --region asia-northeast3

gcloud container clusters get-credentials loopvision-gpu-cluster --region asia-northeast3


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
  http://EXTERNAL_IP/predict # Use the IP you got from kubectl get service








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

