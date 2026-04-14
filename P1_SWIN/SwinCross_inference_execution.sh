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

INFERENCE_OUTPUT_FOLDER=/data/santiago/Datast001_HECKTOR_SwinCross/predictions/inference_220epochs_544patients
MODEL_DIR=hecktor_1gpu_2000ep_run/
LOGGER_NAME=220epochs_544patients

##################################################################
#Step 3 : Run inference
##################################################################
# CUDA_VISIBLE_DEVICES=1 python3.12 test_santiago_manual_reorient.py \
#    --pretrained_dir ./runs/$MODEL_DIR \
#    --pretrained_model_name model_best.pth \
#    --output_dir  $INFERENCE_OUTPUT_FOLDER/local \
#    --data_dir $PPDATA_FOLDER \
#    --json_list dataset_swincross_testing_group.json \
#    --infer_overlap 0.5 \
#    > ./runs/$MODEL_DIR/inference_debug.log 2>&1

CUDA_VISIBLE_DEVICES=1 python3.12 test_monai_invertd_v4M1.py \
                --pretrained_dir ./runs/$MODEL_DIR \
                --pretrained_model_name model_last.pth \
                --output_dir $INFERENCE_OUTPUT_FOLDER \
                --data_dir $PPDATA_FOLDER \
                --json_list dataset_swincross_testing_group.json \
                --infer_overlap 0.5 \
                > ./runs/$MODEL_DIR/inference_debug_$LOGGER_NAME.log 2>&1

##################################################################
#Step 4 : Postprocess and evaluate results
##################################################################
# save graphics of training evolution
# uv run export_graphs.py --logdir ./runs/$MODEL_DIR --output ./runs/$MODEL_DIR/training_graphics/
# uv run export_graphs.py --logdir ./runs/hecktor_1gpu_2000ep_run/ --output ./runs/hecktor_1gpu_2000ep_run/training_graphics/