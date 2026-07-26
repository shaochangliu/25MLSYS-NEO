#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/sliu27/baseline/NEO"
LOG_DIR="$REPO_DIR/logs_largerPP"
VENV_ACTIVATE="$REPO_DIR/.venv/bin/activate"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "[Error] Expected venv activation script at $VENV_ACTIVATE" >&2
    exit 1
fi

source "$VENV_ACTIVATE"
export TORCH_CUDA_ARCH_LIST=9.0

GPU_MEM_UTILIZATION=0.242 # 23GB
SWAP_SPACE=60
MAX_BATCH_SIZE=64
NB=16
PROFILE_DIR="$REPO_DIR/profile_results"
SCRIPT_TIMEOUT_SECONDS=$((6 * 60 * 60))
CURRENT_NEO_PGID=""
WATCHDOG_PID=""

cleanup_current_neo() {
    if [[ -z "${CURRENT_NEO_PGID:-}" ]]; then
        return
    fi

    if kill -0 "$CURRENT_NEO_PGID" 2>/dev/null; then
        echo "[Cleanup] Stopping NEO process group $CURRENT_NEO_PGID"
        kill -TERM "-$CURRENT_NEO_PGID" 2>/dev/null || true
        sleep 10
        kill -KILL "-$CURRENT_NEO_PGID" 2>/dev/null || true
    fi
}

cleanup_on_exit() {
    local status=$?
    if [[ -n "${WATCHDOG_PID:-}" ]]; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    cleanup_current_neo
    exit "$status"
}

timeout_self() {
    sleep "$SCRIPT_TIMEOUT_SECONDS"
    echo "[Timeout] run_NEO.sh exceeded ${SCRIPT_TIMEOUT_SECONDS}s; terminating current NEO run and exiting."
    cleanup_current_neo
    kill -TERM "$$" 2>/dev/null || true
}

trap cleanup_on_exit EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
timeout_self &
WATCHDOG_PID=$!

# Format: "ModelName:HFModelPath:NPP:NTG:NPL"
# NEO num-prompts is NPL * NB to align with llama.cpp's NPL/NB setup.
runs=(
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:256:48"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:256:1024:48"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:512:512:60"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:64:56"
    "llama3.1-8b:/scratch/sliu27/models/HF/llama3.1-8b:1024:32:58"

    # "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:1024:256:14"
    # "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:256:1024:14"
    # "llama2-7b:/scratch/sliu27/models/HF/llama2-7b:512:512:18"
)

clear_profile_results() {
    rm -rf "$PROFILE_DIR"
    mkdir -p "$PROFILE_DIR"
}

run_neo_cmd() {
    local name=$1; local model_path=$2; local npp=$3; local ntg=$4; local npl=$5; local nb=$6

    local num_prompts=$(( npl * nb ))
    local max_tokens_in_batch=$(( npp + ntg ))
    local log_file="${LOG_DIR}/${name}.log"
    local json_file="${LOG_DIR}/${name}.json"

    echo "  -> Executing: $name"
    local full_cmd=(
        python evaluation/batched_bench.py
        --model-path "$model_path"
        --num-prompts "$num_prompts"
        --input-len "$npp"
        --output-len "$ntg"
        --gpu-mem-utilization "$GPU_MEM_UTILIZATION"
        --swap-space "$SWAP_SPACE"
        --max-tokens-in-batch "$max_tokens_in_batch"
        # --max-batch-size "$MAX_BATCH_SIZE"
        --extra-layer-for-cprf
        --record-finish-time-distribution
        --output-json "$json_file"
    )

    printf "     Command:"
    printf " %q" "${full_cmd[@]}"
    printf "\n"

    cd "$REPO_DIR"
    clear_profile_results
    setsid "${full_cmd[@]}" > "$log_file" 2>&1 &
    local neo_pid=$!
    CURRENT_NEO_PGID=$neo_pid

    set +e
    wait "$neo_pid"
    local status=$?
    set -e
    CURRENT_NEO_PGID=""

    if (( status != 0 )); then
        echo "    [Error] $name failed. See $log_file"
        return "$status"
    fi
    clear_profile_results

    sleep 5
}

TOTAL=${#runs[@]}
COUNT=0

for entry in "${runs[@]}"; do
    IFS=':' read -r M_NAME HF_PATH NPP NTG NPL <<< "$entry"

    COUNT=$((COUNT + 1))
    echo "=================================================="
    echo "[$COUNT/$TOTAL] Model: $M_NAME | NPP: $NPP | NTG: $NTG | NPL: $NPL | NB: $NB"
    echo "=================================================="

    PREFIX="${M_NAME}_NPP${NPP}_NTG${NTG}_NPL${NPL}_NB${NB}"
    run_neo_cmd "${PREFIX}_NEO" "$HF_PATH" "$NPP" "$NTG" "$NPL" "$NB"

done

echo "=================================================="
echo "NEO benchmarks completed. Logs saved in $LOG_DIR."
