# AI Model Deployment to GKE

Deploying a GPU-accelerated machine learning model to Google Kubernetes Engine.

This will cost you about $70 USD per day.


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

#### Check available space
df -h
docker system df



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


#### Test locally first:

docker run --rm \
  -e PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "${IMAGE_NAME}" \
  python -c "
import torch
import supervision as sv
import cv2
print('✅ OpenCV version:', cv2.__version__)
print('✅ Supervision version:', sv.__version__)
print('✅ PyTorch version:', torch.__version__)
print('✅ All imports successful!')
"



#### To run the full application FastAPI app locally:
docker run --rm -e PYTORCH_ENABLE_MPS_FALLBACK=1 -p 5050:5050 "${IMAGE_NAME}"


### Test Docker container locally:

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  http://localhost:5050/predict



### Push the image to Artifact Registry
docker push "${IMAGE_NAME}"

see:
https://console.cloud.google.com/artifacts?authuser=3&inv=1&invt=Ab2R_w&project=ai-platform-453201&supportedpurview=project

https://console.cloud.google.com/artifacts/docker/ai-platform-453201/asia-northeast3/docker-ai-model-repo?authuser=2&inv=1&invt=Ab4zTw&project=ai-platform-453201



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

### Create a GKE Cluster (if you don't have one) (VM Instance Group):

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


Confirm that all four nodes in your cluster are in a Ready state. This means your GKE cluster and all its nodes are healthy and available to run pods.



## Create a GPU Node Pool:

--machine-type n1-standard-8 \ # Or a suitable machine type for T4s (e.g., n1-standard-X, g2-standard-X)
  --accelerator "type=nvidia-tesla-t4,count=1" \ # 1 T4 GPU per node
  --num-nodes 1 \ # Start with 1 node, can scale up
  --min-nodes 1 \ # Keep at least one GPU node warm
  --max-nodes 4 \ # Scale up to 4 GPU nodes (total 4 GPUs)
  --enable-autoscaling \
  --node-locations asia-northeast3-a,asia-northeast3-b \ # Should be in zones where T4s are available
  --disk-size 100GB \ # Ensure enough disk space for the large image



### Create the GPU Node Pool with the T4 GPUs:

$ gcloud container node-pools create gpu-node-pool \
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


Check the Node Pool Status:

gcloud container node-pools list --cluster loopvision-gpu-cluster --region asia-northeast3

gcloud container clusters describe loopvision-gpu-cluster --region asia-northeast3


See:
https://console.cloud.google.com/kubernetes/list/overview?authuser=3&inv=1&invt=Ab2UJg&project=ai-platform-453201&supportedpurview=project

https://container.googleapis.com/v1/projects/ai-platform-453201/zones/asia-northeast3/clusters/loopvision-gpu-cluster/nodePools/gpu-node-pool




# Deploy to GKE and Test

Make a k8s-deployment.yaml file to use the correct label that GKE applies to GPU nodes. Get the Correct Node Label for the k8s deployment yaml file.

kubectl get nodes --show-labels


Make sure the GPU nodes have the label: 
 cloud.google.com/gke-accelerator=nvidia-tesla-t4.



# Apply the Kubernetes Manifests

## This will tell Kubernetes to create the pods (deployment):

kubectl apply -f k8s-deployment.yaml

kubectl get service loopvision-gpu-api-service


Once the command completes run:

kubectl get pods 

(see the pod successfully being scheduled and changing its status from Pending to ContainerCreating and finally Running.)


## If you need to re-deploy after modifications:

Tell GKE to pull and use the new image you just pushed:

kubectl set image deployment/loopvision-gpu-api-deployment loopvision-inference-container=asia-northeast3-docker.pkg.dev/ai-platform-453201/docker-ai-model-repo/loopvision-inference-api-gpu:v1


wait for about 10 mins....

kubectl get pods

kubectl get service loopvision-gpu-api-service


## Check Container Logs 
Once the pod is Running, check its logs to confirm the model loaded successfully and the Uvicorn server started.

kubectl logs loopvision-gpu-api-deployment-7f494f8bcb-rr2sq 


# Test Predictions in the endpoint:

curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/Users/oysterable/delete/recyclables-detector/rc40cocodataset_splitted/test/images/5b706395-c141-41d9-8b8c-b7d07a684094.png" \
  http://34.64.193.133/predict


Sample output:

{"detections":[{"box":{"x1":158.37937927246094,"y1":298.0939025878906,"x2":635.03076171875,"y2":575.6695556640625},"confidence":0.9471864700317383,"class_id":1,"label":"PET"}]}





## If errors:

Check Pod Status
kubectl get pods


View the logs for that pod
kubectl logs <POD_NAME>




### Check Service and Load Balancer
If your pod is Running but curl fails, the Load Balancer might still be provisioning.

kubectl get service



# Secure Your API
Your service is currently publicly accessible. For a real-world application, you should add an authentication layer and secure your endpoint. 

You can use Kubernetes Ingress to add SSL certificates and configure domain names.




# Cost Management 

GPU instances are expensive. 

If you don't need the service running continuously, scale your deployment down to zero replicas:

kubectl scale deployment loopvision-gpu-api-deployment --replicas=0


When you need to use the service again, just scale it back up to your desired number of replicas:

kubectl scale deployment loopvision-gpu-api-deployment --replicas=1




# Stop Charges:

## Compute Engine

gcloud compute instances list

gcloud compute instances stop <INSTANCE_NAME> --zone=<ZONE_NAME>


Approx. $1,000 USD / month



## GKE Cluster
GKE clusters and their node pools are a major source of cost. To stop charges without deleting the cluster, you should scale the node pools down to zero nodes. This keeps the cluster's control plane running (which has a small, static cost) but stops all charges for the actual worker machines.

Scale Down Node Pools to Pause
To scale down the gpu-node-pool to 0 nodes:


gcloud container clusters resize loopvision-gpu-cluster \
  --node-pool gpu-node-pool \
  --num-nodes=0 \
  --region asia-northeast3


To reactivate by scaling back up to a specific number of nodes (e.g., 4):

gcloud container clusters resize loopvision-gpu-cluster \
  --node-pool gpu-node-pool \
  --num-nodes=4 \
  --region asia-northeast3



This will provision a new node, and your deployment will automatically get scheduled on it.


Approx. $70 USD / month






# Cloud Run Service
Cloud Run is a serverless platform, so it scales down to zero instances by default when it's not being used. The simplest way to "pause" it and stop all charges is to set the minimum number of instances to zero.

To check the current configuration:

gcloud run services describe loopvision-inference-service --region asia-northeast3

Look for the min-instances setting. To set min-instances to 0 (pause):

gcloud run services update loopvision-inference-service \
  --image asia-northeast3-docker.pkg.dev/ai-platform-453201/docker-ai-model-repo/loopvision-inference-api-gpu:v1 \
  --min-instances=0 \
  --region asia-northeast3


check status again:

gcloud run services describe loopvision-inference-service --region asia-northeast3  



To set min-instances back to 1 or higher (reactivate):

gcloud run services update loopvision-inference-service \
  --min-instances=1 \
  --region asia-northeast3


This will spin up a warm instance again, eliminating the cold start for the next request.


Approx. $25 USD / month




## Load Balancer

to stop:
kubectl delete service loopvision-gpu-api-service

This command will delete the Load Balancer and the external IP address, and the billing for that resource will stop. When you need to access your application again, you can simply re-apply your k8s-deployment.yaml to recreate the service.

kubectl apply -f k8s-deployment.yaml

kubectl get service loopvision-gpu-api-service



External IP (networking) fees:
Approx. $25 USD / month


## Artifact Registry
Artifact Registry is a storage service. It charges based on the amount of data stored and network egress. The only way to stop these charges is to delete the stored images or the repository itself. You can't "pause" a repository.
To delete the entire repository and all images within it:

$ gcloud artifacts repositories delete docker-ai-model-repo --location=asia-northeast3 --quiet

If you do this, you'll need to rebuild and push your Docker image again when you resume.


Approx. $2 USD / month



