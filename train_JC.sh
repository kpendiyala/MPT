#!/bin/bash
set -e

if [[ -z "${DATA_PATH}" ]]; then
    echo "Error: The DATA_PATH environment variable is not set."
    exit 1
fi
if [[ -z "${OUTPUT_PATH}" ]]; then
    echo "Error: The OUTPUT_PATH environment variable is not set."
    exit 1
fi

DATADIR="${DATA_PATH}/JetClass/Pythia"
OUTPUT_VOL_DIR="${OUTPUT_PATH}"

echo "args: $@"

MODEL_NAME=$1
if ! [[ "${MODEL_NAME}" =~ ^(MPT|AuxFreeMPT|PairwiseMPT)$ ]]; then
    echo "Invalid model ${MODEL_NAME}! Valid options: MPT, AuxFreeMPT. PairwiseMPT."
    exit 1
fi
shift

if [[ -z "$1" ]] || [[ "$1" == --* ]]; then
    echo "Error: The second argument must be the feature type (e.g., full, kin, kinpid)."
    exit 1
fi
FEATURE_TYPE=$1
shift

TRAIN_PERCENTAGE=100

WEAVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-percentage) TRAIN_PERCENTAGE="$2"; shift 2 ;;
        *) WEAVER_ARGS+=("$1"); shift ;;
    esac
done

suffix=${COMMENT}
NGPUS=${DDP_NGPUS:-1}

if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS -- $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

NUM_WORKERS=2
if ((TRAIN_PERCENTAGE == 1)); then
    NUM_WORKERS=1
fi

epochs=100
samples_per_epoch=$(((TRAIN_PERCENTAGE * 1000 * 1024) / (10 * NGPUS) ))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers $NUM_WORKERS --fetch-step 0.01"

modelopts="model/${MODEL_NAME}.py --use-amp"
batchopts="--batch-size 512 --start-lr 1e-3"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

mkdir -p "${OUTPUT_VOL_DIR}/training" "${OUTPUT_VOL_DIR}/logs" "${OUTPUT_VOL_DIR}/tensorboard" "${OUTPUT_VOL_DIR}/results"

ln -sfn "${OUTPUT_VOL_DIR}/tensorboard" runs

END_INDEX=$((TRAIN_PERCENTAGE - 1))

expand_files() {
    local category=$1
    local files=()
    for i in $(seq -f "%03g" 0 ${END_INDEX}); do
        files+=("${DATADIR}/train_100M/${category}_${i}.root")
    done
    printf "%s\n" "${files[@]}"
}

$CMD \
    --data-train \
    $(expand_files HToBB | xargs -I {} echo "HToBB:{}") \
    $(expand_files HToCC | xargs -I {} echo "HToCC:{}") \
    $(expand_files HToGG | xargs -I {} echo "HToGG:{}") \
    $(expand_files HToWW2Q1L | xargs -I {} echo "HToWW2Q1L:{}") \
    $(expand_files HToWW4Q | xargs -I {} echo "HToWW4Q:{}") \
    $(expand_files TTBar | xargs -I {} echo "TTBar:{}") \
    $(expand_files TTBarLep | xargs -I {} echo "TTBarLep:{}") \
    $(expand_files WToQQ | xargs -I {} echo "WToQQ:{}") \
    $(expand_files ZToQQ | xargs -I {} echo "ZToQQ:{}") \
    $(expand_files ZJetsToNuNu | xargs -I {} echo "ZJetsToNuNu:{}") \
    --data-val "${DATADIR}/val_5M/*.root" \
    --data-test \
    "HToBB:${DATADIR}/test_20M/HToBB_*.root" \
    "HToCC:${DATADIR}/test_20M/HToCC_*.root" \
    "HToGG:${DATADIR}/test_20M/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/test_20M/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/test_20M/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/test_20M/TTBar_*.root" \
    "TTBarLep:${DATADIR}/test_20M/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/test_20M/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/test_20M/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/test_20M/ZJetsToNuNu_*.root" \
    --data-config dataset/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix ${OUTPUT_VOL_DIR}/training/JetClass/Pythia/${FEATURE_TYPE}/${MODEL_NAME}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log ${OUTPUT_VOL_DIR}/logs/JetClass_Pythia_${FEATURE_TYPE}_${MODEL_NAME}_{auto}${suffix}.log \
    --predict-output ${OUTPUT_VOL_DIR}/results/JetClass_Pythia_${FEATURE_TYPE}_${MODEL_NAME}${suffix}/pred.root \
    --tensorboard JetClass_Pythia_${FEATURE_TYPE}_${MODEL_NAME}${suffix} \
    "${WEAVER_ARGS[@]}"