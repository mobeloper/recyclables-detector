


gcloud compute scp --recurse ~/delete/recyclables-detector/rc40cocodataset_splitted/val/coco_val.json model-training-n1-gpu-nvidia-t4-1:~/ --zone=us-central1-a

gcloud compute scp --recurse ~/delete/recyclables-detector/fine_tuning/config.yaml model-training-n1-gpu-nvidia-t4-1:~/ --zone=us-central1-a

gcloud compute scp --recurse ~/delete/recyclables-detector/Jan2025_ver2_merged_1024_1024 model-training-n1-gpu-nvidia-t4-1:~/ --zone=us-central1-a


Connect to the VM instance:
gcloud compute ssh --project=ai-platform-453201 --zone=us-central1-a model-training-n1-gpu-nvidia-t4-1


Environtment setup:

sudo yum update

sudo yum install -y python3 python3-pip

sudo yum install google-cloud-sdk

sudo yum install wget

sudo yum install jupyter
jupyter notebook --generate-config


gcloud init

gsutil --version


increse disk size:
gcloud compute disks resize model-training-n1-gpu-nvidia-t4-1 --size=50 --zone=us-central1-a

grow data section to the maximum available:
sudo xfs_growfs -d /


Install python:

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

~/miniconda3/bin/conda init bash


==

conda create -n circularnet-train python=3.11

conda activate circularnet-train

df -h


pip install --upgrade setuptools

pip install tensorflow[and-cuda]
pip install tf-models-official
pip install -U -q "tf-models-official"
pip install -q tf-models-nightly

pip show tf-models-official


export TF_FORCE_GPU_ALLOW_GROWTH=true


python -m official.vision.train --mode="train_and_eval" --helpfull | grep "experiment="

grep -r "register_config" ~/miniconda3/envs/circularnet-train/lib/python3.12/site-packages/official/



Download the model weights:

sudo yum install unzip

wget https://storage.googleapis.com/tf_model_garden/vision/waste_identification_ml/Jan2025_ver2_merged_1024_1024.zip

mkdir Jan2025_ver2_merged_1024_1024
unzip Jan2025_ver2_merged_1024_1024.zip -d Jan2025_ver2_merged_1024_1024
rm Jan2025_ver2_merged_1024_1024.zip


https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
Model name: Mask R-CNN Inception ResNet V2 1024x1024

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

tar -xvzf mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz

remove zip file:
rm -rf mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz


Connect to the VM:
gcloud compute ssh --project=ai-platform-453201 --zone=us-central1-a model-training-n1-gpu-nvidia-t4-1


Activate conda env:

conda activate circularnet-train 

nvidia-smi

python -m official.vision.train --experiment="circularnet_finetuning" --mode="train_and_eval" --model_dir="output_directory" --config_file="config.yaml"

python -m official.vision.train --experiment="maskrcnn_resnetfpn_coco" --mode="train_and_eval" --model_dir="output_directory" --config_file="config.yaml"


maskrcnn_mobilenet_coco
maskrcnn_resnetfpn_coco

Hyperparameter tuning optimization was done by changing image size, batch size, learning rate, training steps, epochs and data augmentation steps


===
Start Jupyter Notebook
Run:
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

Copy the token from the output and access Jupyter at:
http://<your-vm-external-ip>:8888

optional:
nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser > jupyter.log 2>&1 &
