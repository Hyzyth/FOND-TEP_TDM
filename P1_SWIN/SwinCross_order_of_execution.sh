# This file makes part of the  Optimization of prognostic factors 
# for H&N cancer treatment through longitudinal analysis of PET/CT data (LongiTEP).
# Coded by : Santiago
# Anotated by : Santiago
# file creation date : 29 jan 2026

# Summary :This file contains the bash execution commands to prepare the data, train and infer a swincross unetr model
# for medical tumour segmentation task, rebuilt by M1 students Elias, Alex and Paul-André
#using the HECKTOR dataset.

# Current dataset : Dataset_001_HECKTORct

#Step 0 : activate swincross envionment and check requirements
# check uv installation and install python3.12 if not present
if ! command -v uv &> /dev/null
then
    echo "uv could not be found, installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
fi

uv python install 3.12

# create and activate swincross virtual environment
if [ ! -d "swincross_env" ]; then
    python3.12 -m venv swincross_env
fi

source swincross_env/bin/activate

requirements_path=requirements.txt

if [ -f $requirements_path ]; then
    uv pip install -r $requirements_path
else
    echo "Requirements file not found!"
fi

PPDATA_FOLDER=/data/santiago/Datast001_HECKTOR_SwinCross/
RAWDATA_FOLDER=/home/santiago/HECKTOR_data/Task_1_segmentation/
INFERENCE_OUTPUT_FOLDER=/data/santiago/Datast001_HECKTOR_SwinCross_2000ep_res/
MODEL_DIR=hecktor_1gpu_2000ep_run/ 

##################################################################
# #Step 1 : Preprocess HECKTOR dataset to swincross format
##################################################################
# uv run dataset_builder_simpleITK.py --input_folder $RAWDATA_FOLDER \
#                                    --output_folder $PPDATA_FOLDER --val_split 0.2 --seed 42 # Do not change for reproducibility

##################################################################
#Step 2 : Train the model
##################################################################
#use both GPU 0 and 1 for training
# Note : Multiple GPU training currently not working. Try adding huggingface accelerator lib if later use is necessary
# CUDA_VISIBLE_DEVICES=0,1 torchrun \
#     --nproc_per_node=2 \
#     train.py \
#     --data_dir $PPDATA_FOLDER \
#     --distributed --batch_size 2 \
#     --max_epochs 2000 \
#     --val_every 50 \
#     --warmup_epochs 100 \
#     --workers 6 \
#     --cache_rate 0.5 \
#     --save_checkpoint \
#     --logdir $MODEL_DIR \

#use only GPU 0 for training
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3.12 -u train.py \
    --data_dir $PPDATA_FOLDER \
    --batch_size 2 \
    --max_epochs 2000 \
    --val_every 20 \
    --warmup_epochs 25 \
    --workers 4 \
    --cache_rate 1.0 \
    --save_checkpoint \
    --logdir $MODEL_DIR \
   > ./runs/$MODEL_DIR/training_debug.log 2>&1 # redirecting stdout and stderr to a log file. excellent if model crashes

# testing single GPU training for error handling with all HECKTOR data
# CUDA_VISIBLE_DEVICES=0 python3.12 train.py \
#     --data_dir $PPDATA_FOLDER \
#     --batch_size 2 \
#     --max_epochs 5 \
#     --val_every 1 \
#     --warmup_epochs 1 \
#     --workers 0 \
#     --cache_rate 0.0 \
#     --logdir test_single_gpu \

##################################################################
#Step 3 : Run inference
##################################################################
# CUDA_VISIBLE_DEVICES=1 python3.12 test.py \
#    --pretrained_dir ./runs/$MODEL_DIR \
#    --pretrained_model_name model_best.pth \
#    --output_dir  $INFERENCE_OUTPUT_FOLDER \
#    --data_dir $PPDATA_FOLDER \
#    --json_list dataset_swincross.json \
#    --infer_overlap 0.5

#testing inference 
# CUDA_VISIBLE_DEVICES=1 python3.12 test.py  --pretrained_dir ./runs/test_debug/  --pretrained_model_name model_best.pth --output_dir /data/santiago/test_debug/  --data_dir ./Dataset_Final_SwinCross_SITK/ --json_list dataset_swincross.json  --infer_overlap 0.5

##################################################################
#Step 4 : Postprocess and evaluate results
##################################################################
# save graphics of training evolution
# uv run export_graphics.py --logdir ./runs/$MODEL_DIR --output_dir ./runs/$MODEL_DIR/training_graphics/