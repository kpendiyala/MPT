#!/bin/bash
set -euo pipefail

# JetClass-II training wrapper for Haris's MPT/MoEParT models.
#
# Expected environment:
#   DATA_PATH=/kaushik-moe-vol/data
#   OUTPUT_PATH=/kaushik-moe-vol/outputs
#   COMMENT=debug-run-name
#   DDP_NGPUS=1
#
# Expected data layout:
#   ${DATA_PATH}/JetClassII/Pythia/Res2P_0000.parquet
#   ${DATA_PATH}/JetClassII/Pythia/Res34P_0000.parquet
#   ${DATA_PATH}/JetClassII/Pythia/QCD_0000.parquet
#
# Usage examples:
#   ./train_JC2.sh MPT full --smoke-test --dry-run
#   ./train_JC2.sh MPT full --smoke-test --network-option moe_num_experts 4 --network-option moe_top_k 1
#   ./train_JC2.sh MPT full --train-files-per-group 10 --val-files-per-group 2
#   ./train_JC2.sh MPT full --full-dataset

if [[ -z "${DATA_PATH:-}" ]]; then
    echo "Error: The DATA_PATH environment variable is not set."
    exit 1
fi
if [[ -z "${OUTPUT_PATH:-}" ]]; then
    echo "Error: The OUTPUT_PATH environment variable is not set."
    exit 1
fi

DATADIR="${DATA_PATH}/JetClassII"
OUTPUT_VOL_DIR="${OUTPUT_PATH}"

echo "args: $@"
echo "DATADIR=${DATADIR}"
echo "OUTPUT_VOL_DIR=${OUTPUT_VOL_DIR}"

MODEL_NAME=${1:-}
if ! [[ "${MODEL_NAME}" =~ ^(ParT|MPT|AuxFreeMPT|PairwiseMPT)$ ]]; then
    echo "Invalid model ${MODEL_NAME}! Valid options: ParT, MPT, AuxFreeMPT, PairwiseMPT."
    exit 1
fi
shift

if [[ -z "${1:-}" ]] || [[ "${1:-}" == --* ]]; then
    echo "Error: The second argument must be the feature type. For JetClass-II, use: full"
    exit 1
fi
FEATURE_TYPE=$1
shift

# For now we only have the Sophon JetClass-II full config.
# Later, if you create JetClassII_kin.yaml / JetClassII_kinpid.yaml, this can be expanded.
if [[ "${FEATURE_TYPE}" != "full" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}! For the current JetClass-II config, use: full"
    exit 1
fi

# Defaults are intentionally small/safe.
MODE="train"
SMOKE_TEST=0
FULL_DATASET=0
DRY_RUN=0
TRAIN_FILES_PER_GROUP=1
VAL_FILES_PER_GROUP=1
EPOCHS=2
NUM_WORKERS=2
BATCH_SIZE=512
START_LR="5e-4"
SAMPLES_PER_EPOCH=""
SAMPLES_PER_EPOCH_VAL=""

WEAVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --smoke-test)
            SMOKE_TEST=1
            TRAIN_FILES_PER_GROUP=1
            VAL_FILES_PER_GROUP=1
            EPOCHS=2
            NUM_WORKERS=1
            shift ;;
        --full-dataset)
            FULL_DATASET=1
            EPOCHS=80
            NUM_WORKERS=5
            shift ;;
        --train-files-per-group)
            TRAIN_FILES_PER_GROUP="$2"; shift 2 ;;
        --val-files-per-group)
            VAL_FILES_PER_GROUP="$2"; shift 2 ;;
        --num-epochs)
            EPOCHS="$2"; shift 2 ;;
        --num-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --start-lr)
            START_LR="$2"; shift 2 ;;
        --samples-per-epoch)
            SAMPLES_PER_EPOCH="$2"; shift 2 ;;
        --samples-per-epoch-val)
            SAMPLES_PER_EPOCH_VAL="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        # Backward-compatible alias from the old JetClass script.
        # For JetClass-II this means "number of files per group", not literal percentage.
        --train-percentage)
            TRAIN_FILES_PER_GROUP="$2"; shift 2 ;;
        *)
            WEAVER_ARGS+=("$1"); shift ;;
    esac
done

if ! [[ "${MODE}" =~ ^(make_weight|train|convert)$ ]]; then
    echo "Invalid mode ${MODE}! Valid options: make_weight, train, convert."
    exit 1
fi

suffix=${COMMENT:-debug}
NGPUS=${DDP_NGPUS:-1}

if ((NGPUS > 1)); then
    CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS}" "$(which weaver)" --backend nccl)
else
    CMD=(weaver)
fi

# Sophon/JetClass-II defaults.
samples_per_epoch=$((10000 * 1024 / NGPUS))
samples_per_epoch_val=$((2500 * 1024))

if (( SMOKE_TEST == 1 )); then
    # Keep smoke tests short and cheap.
    samples_per_epoch=$((20 * 1024 / NGPUS))
    samples_per_epoch_val=$((10 * 1024))
fi

# Optional explicit limits for pilot runs.
if [[ -n "${SAMPLES_PER_EPOCH}" ]]; then
    samples_per_epoch=$((SAMPLES_PER_EPOCH / NGPUS))
fi

if [[ -n "${SAMPLES_PER_EPOCH_VAL}" ]]; then
    samples_per_epoch_val="${SAMPLES_PER_EPOCH_VAL}"
fi

DATACONFIG="data/JetClassII/JetClassII_full.yaml"
NETWORK_CONFIG="model/${MODEL_NAME}.py"

DATAOPTS=(--num-workers "${NUM_WORKERS}" --fetch-step 1.0 --data-split-num 200)
BATCHOPTS=(--batch-size "${BATCH_SIZE}" --start-lr "${START_LR}")

mkdir -p \
    "${OUTPUT_VOL_DIR}/training" \
    "${OUTPUT_VOL_DIR}/logs/${MODEL_NAME}" \
    "${OUTPUT_VOL_DIR}/tensorboard" \
    "${OUTPUT_VOL_DIR}/results/${MODEL_NAME}" \
    "${OUTPUT_VOL_DIR}/experiments"

ln -sfn "${OUTPUT_VOL_DIR}/tensorboard" runs

train_files=()
val_files=()
weight_files=()

add_labeled_range() {
    local -n arr_ref=$1
    local label=$2
    local prefix=$3
    local start=$4
    local end=$5

    for i in $(seq -w "${start}" "${end}"); do
        arr_ref+=("${label}:${DATADIR}/Pythia/${prefix}_${i}.parquet")
    done
}

add_unlabeled_range() {
    local -n arr_ref=$1
    local prefix=$2
    local start=$3
    local end=$4

    for i in $(seq -w "${start}" "${end}"); do
        arr_ref+=("${DATADIR}/Pythia/${prefix}_${i}.parquet")
    done
}

add_labeled_n_from_start() {
    local -n arr_ref=$1
    local label=$2
    local prefix=$3
    local start_int=$4
    local n=$5

    local end_int=$((start_int + n - 1))
    for i in $(seq "${start_int}" "${end_int}"); do
        arr_ref+=("${label}:${DATADIR}/Pythia/${prefix}_$(printf "%04d" "${i}").parquet")
    done
}

add_unlabeled_n_from_start() {
    local -n arr_ref=$1
    local prefix=$2
    local start_int=$3
    local n=$4

    local end_int=$((start_int + n - 1))
    for i in $(seq "${start_int}" "${end_int}"); do
        arr_ref+=("${DATADIR}/Pythia/${prefix}_$(printf "%04d" "${i}").parquet")
    done
}

if (( FULL_DATASET == 1 )); then
    # Official Sophon/JetClass-II ranges.
    # Training: Res2P 0000-0199, Res34P 0000-0859, QCD 0000-0279
    # Val:      Res2P 0200-0249, Res34P 0860-1074, QCD 0280-0349
    # Weights:  train+val ranges.
    add_labeled_range train_files  Res2P  Res2P  0000 0199
    add_labeled_range train_files  Res34P Res34P 0000 0859
    add_labeled_range train_files  QCD    QCD    0000 0279

    add_unlabeled_range val_files Res2P  0200 0249
    add_unlabeled_range val_files Res34P 0860 1074
    add_unlabeled_range val_files QCD    0280 0349

    add_unlabeled_range weight_files Res2P  0000 0249
    add_unlabeled_range weight_files Res34P 0000 1074
    add_unlabeled_range weight_files QCD    0000 0349
else
    # Small/subset mode.
    # Training starts at official training offsets.
    # Validation starts at official validation offsets.
    add_labeled_n_from_start train_files Res2P  Res2P  0   "${TRAIN_FILES_PER_GROUP}"
    add_labeled_n_from_start train_files Res34P Res34P 0   "${TRAIN_FILES_PER_GROUP}"
    add_labeled_n_from_start train_files QCD    QCD    0   "${TRAIN_FILES_PER_GROUP}"

    add_unlabeled_n_from_start val_files Res2P  200 "${VAL_FILES_PER_GROUP}"
    add_unlabeled_n_from_start val_files Res34P 860 "${VAL_FILES_PER_GROUP}"
    add_unlabeled_n_from_start val_files QCD    280 "${VAL_FILES_PER_GROUP}"

    # For a tiny smoke test, use the available tiny train+val subset for weight calculation if needed.
    weight_files+=("${val_files[@]}")
    for f in "${train_files[@]}"; do
        weight_files+=("${f#*:}")
    done
fi

check_files_exist() {
    local missing=0
    for item in "$@"; do
        local path="${item#*:}"  # strips optional label prefix
        if [[ ! -f "${path}" ]]; then
            echo "Missing file: ${path}"
            missing=1
        fi
    done
    if (( missing == 1 )); then
        echo "Error: Some required JetClass-II files are missing."
        echo "Download the needed files into: ${DATADIR}/Pythia"
        exit 1
    fi
}

if [[ "${MODE}" == "train" ]]; then
    check_files_exist "${train_files[@]}" "${val_files[@]}"
elif [[ "${MODE}" == "make_weight" ]]; then
    check_files_exist "${weight_files[@]}"
fi

RUN_NAME="JetClassII_Pythia_${FEATURE_TYPE}_${MODEL_NAME}_${suffix}"
MODEL_PREFIX="${OUTPUT_VOL_DIR}/training/JetClassII/Pythia/${FEATURE_TYPE}/${MODEL_NAME}/${suffix}/net"
LOG_FILE="${OUTPUT_VOL_DIR}/logs/${MODEL_NAME}/${RUN_NAME}.log"
PRED_OUT="${OUTPUT_VOL_DIR}/results/${MODEL_NAME}/${RUN_NAME}/pred.root"

echo "MODE=${MODE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "FEATURE_TYPE=${FEATURE_TYPE}"
echo "RUN_NAME=${RUN_NAME}"
echo "NGPUS=${NGPUS}"
echo "EPOCHS=${EPOCHS}"
echo "TRAIN_FILES=${#train_files[@]}"
echo "VAL_FILES=${#val_files[@]}"
echo "DATACONFIG=${DATACONFIG}"
echo "NETWORK_CONFIG=${NETWORK_CONFIG}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "SAMPLES_PER_EPOCH=${samples_per_epoch}"
echo "SAMPLES_PER_EPOCH_VAL=${samples_per_epoch_val}"
echo "TRAIN_STEPS_APPROX=$(((samples_per_epoch + BATCH_SIZE - 1) / BATCH_SIZE))"
echo "VAL_STEPS_APPROX=$(((samples_per_epoch_val + BATCH_SIZE - 1) / BATCH_SIZE))"

if [[ "${MODE}" == "make_weight" ]]; then
    FINAL_CMD=(
        "${CMD[@]}"
        --print
        --data-train "${weight_files[@]}"
        --data-config "${DATACONFIG}"
        --network-config "${NETWORK_CONFIG}"
        --use-amp --amp-dtype bf16
        -o num_classes 188
        -o fc_params "[(512,0.1)]"
        --model-prefix "${MODEL_PREFIX}"
        "${DATAOPTS[@]}" "${BATCHOPTS[@]}"
        --samples-per-epoch "${samples_per_epoch}"
        --samples-per-epoch-val "${samples_per_epoch_val}"
        --num-epochs "${EPOCHS}"
        --optimizer ranger
        --gpus 0
        --log "${OUTPUT_VOL_DIR}/logs/${MODEL_NAME}/${RUN_NAME}_make_weight.log"
        "${WEAVER_ARGS[@]}"
    )
elif [[ "${MODE}" == "train" ]]; then
    FINAL_CMD=(
        "${CMD[@]}"
        --no-remake-weights
        --data-train "${train_files[@]}"
        --data-val "${val_files[@]}"
        --data-config "${DATACONFIG}"
        --network-config "${NETWORK_CONFIG}"
        --use-amp --amp-dtype bf16
        -o num_classes 188
        -o fc_params "[(512,0.1)]"
        --model-prefix "${MODEL_PREFIX}"
        "${DATAOPTS[@]}" "${BATCHOPTS[@]}"
        --samples-per-epoch "${samples_per_epoch}"
        --samples-per-epoch-val "${samples_per_epoch_val}"
        --num-epochs "${EPOCHS}"
        --optimizer ranger
        --gpus 0
        --log "${LOG_FILE}"
        --predict-output "${PRED_OUT}"
        --tensorboard "${MODEL_NAME}/${RUN_NAME}"
        "${WEAVER_ARGS[@]}"
    )
else
    FINAL_CMD=(
        "${CMD[@]}"
        --no-remake-weights
        --data-config "${DATACONFIG}"
        --network-config "${NETWORK_CONFIG}"
        --use-amp --amp-dtype bf16
        -o num_classes 188
        -o fc_params "[(512,0.1)]"
        -o export_embed True
        --model-prefix "${MODEL_PREFIX}_best_epoch_state.pt"
        --export-onnx "${OUTPUT_VOL_DIR}/results/${MODEL_NAME}/${RUN_NAME}/model.onnx"
        "${WEAVER_ARGS[@]}"
    )
fi

echo "Command:"
printf ' %q' "${FINAL_CMD[@]}"
echo

if (( DRY_RUN == 1 )); then
    echo "Dry run requested; exiting without launching weaver."
    exit 0
fi

"${FINAL_CMD[@]}"
