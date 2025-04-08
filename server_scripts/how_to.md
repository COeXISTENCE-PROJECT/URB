# Steps

1. Navigate to your directory by running `cd /home/$USER/` in your terminal.
2. Clone the project via HTTPS **right here** and not into any subfolder, by running `git clone https://github.com/COeXISTENCE-PROJECT/URB.git` for the main branch.
3. In your terminal, run `cd /home/$USER/URB/server_scripts/` to navigate to this folder.
4. SBatch scripts are already configured. However, you may adjust configurations in `run_simulation.sh` for more resource/shorter wait time.
5. Adjust line 8 of `cmd_container.sh` according to the experiment you want to run.
5. In your terminal, run `sbatch run.sh` to start running the simulation.
6. Outputs of the SLURM job (`slurm-<job_id>.out`) and the consols prints of the program (`container_printouts/output_<job_id>.txt`) will be in this folder.
7. Just be patient, it is possible that the content of `output_<job_id>.txt` is not updated frequently. Check the status of your job using `squeue --me` in your terminal. Or if you suspect the resources you requested are not actually allocated, you can inspect using `scontrol show jobid -d <job_id>`.
8. Once the simulation ends, all records, plots, .csv files that the simulation normally saves to the disk will be accessible in `records` and `plots` folders as usual.