#!/bin/bash -l

#SBATCH --job-name=yolo-distill
#SBATCH --time=999:00:00
#SBATCH --nodelist=nv178
##SBATCH -p 80g
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --mem=160G
#SBATCH --qos=normal
#SBATCH --cpus-per-task=13
#SBATCH -o ./logs/output_%A_%a.txt

export CONTAINER_IMAGE_PATH='/purestorage/project/tyk/0_Software/autodistill.sqsh'
export CACHE_FOL_PATH='/purestorage/project/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/project/tyk/3_CUProjects/Distillation'


srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOL_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    bash -c 'python /purestorage/project/tyk/3_CUProjects/Distillation/roboflow/auto-label.py'