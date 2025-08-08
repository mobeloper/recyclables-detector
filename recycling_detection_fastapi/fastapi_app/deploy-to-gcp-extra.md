
To save predictions from a deployed GKE model, you must use a cloud-native solution for persistent storage, such as Google Cloud Storage (GCS). This is a robust and scalable way to store your images and annotations.

Here is the step-by-step guide to save your predictions to a GCS bucket.

1. Set up a GCS Bucket and Service Account
You only need to do this once.

Create a GCS Bucket: In the Google Cloud Console, go to Cloud Storage > Buckets and create a new bucket. Let's call it your-predictions-bucket. Make sure the location and other settings are appropriate for your needs.

Create a Service Account for GKE: Go to IAM & Admin > Service Accounts and create a new service account (e.g., gke-predictions-writer).

Grant Permissions: Give this service account the Storage Admin or Storage Object Creator role. This is critical as it gives your GKE pod the necessary permissions to write files to the GCS bucket.


==

The GKE documentation recommends using Workload Identity as the modern and secure way to grant pods access to Google Cloud resources.

Your k8s-deployment.yaml manifest already includes the serviceAccountName: gke-predictions-writer field, but to make this work, you need to create and configure that service account to have the necessary permissions.

Here's the step-by-step guide to set up Workload Identity so your pod can write to your GCS bucket.

1. Create a Kubernetes Service Account
First, create a Kubernetes Service Account in your cluster. This will be the identity that your pod uses inside the cluster.

Bash

kubectl create serviceaccount gke-predictions-writer
This command creates a new ServiceAccount object named gke-predictions-writer.

2. Create an IAM Service Account
Next, create an IAM Service Account in your Google Cloud project. This will be the identity that your pod uses to access Google Cloud APIs outside the cluster.

Bash

gcloud iam service-accounts create gke-predictions-writer \
  --display-name="GKE Predictions Writer"
This command creates a new IAM service account with a human-readable display name.

3. Grant IAM Permissions
Now, grant the IAM Service Account the necessary permissions to write to your GCS bucket.

Bash

gcloud projects add-iam-policy-binding ai-platform-453201 \
  --member="serviceAccount:gke-predictions-writer@ai-platform-453201.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
This command gives your new service account the Storage Admin role, allowing it to read and write to your GCS buckets.

4. Establish the Workload Identity Binding
This is the most critical step. You need to create a binding between your Kubernetes Service Account and your IAM Service Account. This tells GKE that the pod using the gke-predictions-writer service account can "impersonate" the gke-predictions-writer IAM service account to access Google Cloud resources.

Bash

gcloud iam service-accounts add-iam-policy-binding gke-predictions-writer@ai-platform-453201.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:ai-platform-453201.svc.id.goog[default/gke-predictions-writer]"
default is the namespace where your service account is created.

gke-predictions-writer is the name of your Kubernetes service account.

5. Annotate the Kubernetes Service Account
You must add an annotation to the Kubernetes Service Account that links it to the IAM Service Account.

Bash

kubectl annotate serviceaccount gke-predictions-writer \
  iam.gke.io/gcp-service-account=gke-predictions-writer@ai-platform-453201.iam.gserviceaccount.com
6. Update the Deployment Manifest
Your k8s-deployment.yaml manifest already has the correct serviceAccountName, but you should apply it again to ensure the deployment uses the newly configured service account.

Bash

kubectl apply -f k8s-deployment.yaml
After these steps, your pod will be able to write to your GCS bucket without needing to handle private keys, which is a much more secure and manageable approach.