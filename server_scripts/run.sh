#!/bin/bash
#SBATCH --job-name=urb_1
#SBATCH --qos=big
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx

PATH_PROGRAM="/home/$USER/URB"
PUT_PROGRAM_TO="/app"
PATH_SUMO_CONTAINER="/shared/sets/singularity/sumo.sif"
CMD_PATH="/home/$USER/URB/server_scripts/cmd_container.sh"
PRINTS_SAVE_PATH="/home/$USER/URB/server_scripts/container_printouts/output_$SLURM_JOB_ID.txt"

mkdir -p "$(dirname "$PRINTS_SAVE_PATH")"
# Run container by adding code by binding, run commands from cmd_container.sh, save printouts to a file
singularity exec --nv --bind "$PATH_PROGRAM":"$PUT_PROGRAM_TO" "$PATH_SUMO_CONTAINER" /bin/bash "$CMD_PATH" > "$PRINTS_SAVE_PATH" 2>&1
