#!/bin/bash
#SBATCH --account=bebv-delta-gpu
#SBATCH --job-name=sbatch_run_pytorch
#SBATCH --output=sbatch_run_pytorch_%j.out
#SBATCH --partition=gpuH200x8
##SBATCH --reservation=affinity1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=04:00:00
##SBATCH --time=36:00:00
##SBATCH --constraint=perf,nvperf

set -ex

cd $SLURM_SUBMIT_DIR
CONTAINER_HOME=$SLURM_SUBMIT_DIR
SIF_FILE=pytorch.sif

ITERS=251
WAIT=9
GRAD_ACCUMLATE_PRE_STEPS=99999
HN=$(hostname -s)

run_job() {
        BS=$1
        CL=$2
        PYTORCH_PROFILE=$3

        CONFIG_FILE="configs/llama-3.1-8b-bs${BS}-${CL}k.json"
        [[ -e $CONFIG_FILE ]] || { 1>&2 echo "Couldn't find config file" 1>&2; exit 1; }
        echo "running b${BS}s${CL}" 1>&2
        [[ -d ${HN}/b${BS}s${CL} ]] || mkdir -p ${HN}/b${BS}s${CL}

        APPTAINER_ARGS=(
                apptainer exec
                --nv
                "${CONTAINER_HOME}/${SIF_FILE}"
        )


        TORCHRUN_ARGS=(
                torchrun
                --nnodes=1
                --node_rank=0
                --nproc_per_node=8
                --master_addr="0.0.0.0"
                --master_port="12234"
                ./train_fsdp.py
                $CONFIG_FILE
                llama
                --use_pytorch_profiler "$PYTORCH_PROFILE"
                --num_iteration "$ITERS"
                --grad_accumlate_pre_steps "$GRAD_ACCUMLATE_PRE_STEPS"
                --output_dir "${HN}/b${BS}s${CL}"
                --wait $WAIT
                --use_fsdp2=1
        )

        echo "RUNANDTIME_START $(date +%s)"
        "${APPTAINER_ARGS[@]}" "${COUNTERS_ARGS[@]}" "${TORCHRUN_ARGS[@]}"
        echo "RUNANDTIME_STOP $(date +%s)"
}

run_job 1 4 1
run_job 2 4 1
run_job 1 8 1

