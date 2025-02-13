#!/bin/bash

#SBATCH --job-name=ultralytics
#SBATCH --partition=80g
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH -o ./logs/output_%A_%a.txt
##SBATCH --container-image=/purestorage/project/tyk/3_CUProjects/ultralytics-syshin/script/ultralytics.sqsh


#SBATCH --container-image=/purestorage/project/tyk/0_Software/autodistill3.sqsh

##SBATCH --container-image=/purestorage/project/tyk/0_Software/yolo_distil.sqsh # head까지 깔림

#SBATCH --container-mounts=/purestorage:/purestorage
##SBATCH --container-mounts=/purestorage/project/tyk/0_Software/cache:/home/$USER/.cache
#SBATCH --container-workdir=/purestorage/project/tyk/3_CUProjects/Distillation
#SBATCH --container-remap-root
#SBATCH --container-writable

unset RANK
unset LOCAL_RANK

pip install torchinfo
pip uninstall ultralytics -y
pip list
python just_yolo2.py
# python down_coco.py