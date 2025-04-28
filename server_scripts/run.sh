#!/bin/bash
#SBATCH --job-name=urb
#SBATCH --qos=big
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx2080

FOLDER_NAME="URB"
PATH_PROGRAM="/home/$USER/$FOLDER_NAME"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo2.sif"
CMD_PATH="$PATH_PROGRAM/server_scripts/cmd_container.sh"
PRINTS_SAVE_PATH="$PATH_PROGRAM/server_scripts/container_printouts/output_$SLURM_JOB_ID.txt"

mkdir -p "$(dirname "$PRINTS_SAVE_PATH")"
# Run container by adding code by binding, run commands from cmd_container.sh, save printouts to a file
singularity exec --nv --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" > "$PRINTS_SAVE_PATH" 2>&1
