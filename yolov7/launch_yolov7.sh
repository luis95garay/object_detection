#!/bin/bash

# Set variables (Required)
dataset="Yolov7Dataset"
model_type="yolov7" # options: yolov7-tiny or yolov7
experiment_name="ElvisTechGo"
run_tag="yolov7_640"

# general hyperparameters (Required)
batch_size=16
epochs=1
img_size=640
data="data.yaml" 

# if training yolov7, complete next hyperparameters (Optional)
cfg_yolov7="yolov7.yaml"
hyp_yolov7="hyp.scratch.custom.yaml"

# if training yolov7-tiny, complete next hyperparameters (Optional)
cfg_yolov7_tiny="yolov7-tiny.yaml"
hyp_yolov7_tiny="hyp.scratch.tiny.yaml"



# Download the official YOLOv7 repo
if [ ! -d  "yolov7" ]; then
    git clone "https://github.com/WongKinYiu/yolov7.git" 
else
    echo "The repository already exists in the target directory."
fi

# Install required packages
pip install -r yolov7/requirements.txt
pip install mlflow
pip install boto3
pip install python-dotenv


# Move modified traninig files
if [ -e "train_mlflow.py" ]; then
    mv train_mlflow.py yolov7
fi
if [ -e "test_mlflow.py" ]; then
    mv test_mlflow.py yolov7
fi


# Download dataset from s3 bucket
if [ ! -d $dataset ]; then
    python3 load_dataset.py \
            --bucket_name elvis-s3-mlflow \
            --dataset_path dataset/$dataset.zip \
            --save_dir $dataset.zip
    
    unzip $dataset.zip -d .
    rm $dataset.zip

else
    echo "The dataset already exist"
fi



if [ "$model_type" = "yolov7" ]; then
    # Download pretrained weights
    cd yolov7
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    cd ..

    # Train an test yolov7 model
    python3 yolov7/train_mlflow.py \
            --device 0 --batch-size $batch_size --epochs $epochs --img $img_size \
            --data $data \
            --cfg $cfg_yolov7 \
            --weights 'yolov7/yolov7_training.pt' \
            --name $run_tag \
            --hyp $hyp_yolov7 \
            --experiment-name $experiment_name

elif [ "$model_type" = "yolov7-tiny" ]; then
    # Download pretrained weights
    cd yolov7
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
    cd ..

    # Train and test yolov7tiny model
    python3 yolov7/train_mlflow.py \
            --device 0 --batch-size $batch_size --epochs $epochs --img $img_size \
            --data $data \
            --cfg $cfg_yolov7_tiny \
            --weights 'yolov7/yolov7-tiny.pt' \
            --name $run_tag \
            --hyp $hyp_yolov7_tiny \
            --experiment-name $experiment_name

fi
