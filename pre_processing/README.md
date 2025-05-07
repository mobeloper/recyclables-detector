
### Move TF dataset to the VM instance

gcloud auth login

gcloud config set project ai-platform-453201

Connect to the VM instance:
gcloud compute ssh --project=ai-platform-453201 --zone=us-central1-a model-training-n1-gpu-nvidia-t4-1

Create the necessary directories: mkdir -p ~/rc40tfrecords
or
Create the necessary directories: mkdir -p ~/rc40tfrecords/val
Create the necessary directories: mkdir -p ~/rc40tfrecords/train

Exit the SSH session: exit

Open the terminal and run the following commands:

Move dataset:
gcloud compute scp --recurse ~/delete/recyclables-detector/rc40tfrecords model-training-n1-gpu-nvidia-t4-1:~/rc40tfrecords --zone=us-central1-a
Or
gcloud compute scp --recurse ~/delete/recyclables-detector/rc40tfrecords/val model-training-n1-gpu-nvidia-t4-1:~/rc40tfrecords --zone=us-central1-a
and
gcloud compute scp --recurse ~/delete/recyclables-detector/rc40tfrecords/train model-training-n1-gpu-nvidia-t4-1:~/rc40tfrecords --zone=us-central1-a





