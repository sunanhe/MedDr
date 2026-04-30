set -x

CHECKPOINT=Sunanhe/MedDr_0401

DATASET=pcam200

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "CHECKPOINT: ${CHECKPOINT}"

MASTER_PORT=${MASTER_PORT:-50924}
PORT=${PORT:-50924}
GPUS=${GPUS:-4}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    inference_gsco.py --checkpoint ${CHECKPOINT} --datasets ${DATASET}