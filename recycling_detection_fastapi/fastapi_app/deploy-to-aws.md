
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

