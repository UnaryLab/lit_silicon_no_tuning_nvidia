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

ITERS="${ITERS:-251}"
WAIT="${WAIT:-9}"
ACTIVE="${ACTIVE:-1}"
GRAD_ACCUMLATE_PRE_STEPS="${GRAD_ACCUMLATE_PRE_STEPS:-99999}"
POWER_MAN="${POWER_MAN:-0}"
ADJUST_STEPS="${ADJUST_STEPS:-3}"
WAIT_STEPS="${WAIT_STEPS:-50}"
INITIAL_POWER_CAP="${INITIAL_POWER_CAP:-750}"
POWER_BUDGET="${POWER_BUDGET:-0}"
REALLOC_POWER="${REALLOC_POWER:-0}"
MAX_ADJ="${MAX_ADJ:-15}"
MAX_POWER="${MAX_POWER:-750}"
USE_SUM="${USE_SUM:-1}"
USE_LAST="${USE_LAST:-0}"
USE_MAX="${USE_MAX:-0}"
USE_GLOBAL="${USE_GLOBAL:-1}"
GRPC_SOCKET="${GRPC_SOCKET:-/tmp/freq.sock}"
HN=$(hostname -s)

cleanup() {
        if [[ -n "${FREQ_SERVER_PID:-}" ]]; then
                kill "$FREQ_SERVER_PID" 2>/dev/null || true
                wait "$FREQ_SERVER_PID" 2>/dev/null || true
        fi
}
trap cleanup EXIT

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
        )
        if [[ "$POWER_MAN" == "1" ]]; then
                APPTAINER_ARGS+=(--bind "${GRPC_SOCKET}:${GRPC_SOCKET}")
        fi
        APPTAINER_ARGS+=("${CONTAINER_HOME}/${SIF_FILE}")


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
                --active "$ACTIVE"
                --power_man "$POWER_MAN"
                --adjust_steps "$ADJUST_STEPS"
                --wait_steps "$WAIT_STEPS"
                --initial_power_cap "$INITIAL_POWER_CAP"
                --power_budget "$POWER_BUDGET"
                --realloc_power "$REALLOC_POWER"
                --max_adj "$MAX_ADJ"
                --max_power "$MAX_POWER"
                --use_sum "$USE_SUM"
                --use_last "$USE_LAST"
                --use_max "$USE_MAX"
                --use_global "$USE_GLOBAL"
                --grpc_socket "$GRPC_SOCKET"
                --use_fsdp2=1
        )

        echo "RUNANDTIME_START $(date +%s)"
        "${APPTAINER_ARGS[@]}" "${COUNTERS_ARGS[@]}" "${TORCHRUN_ARGS[@]}"
        echo "RUNANDTIME_STOP $(date +%s)"
}

if [[ "$POWER_MAN" == "1" ]]; then
        ./freq_server.py &
        FREQ_SERVER_PID=$!
        sleep 3
fi

run_job 1 4 1
run_job 2 4 1
run_job 1 8 1
