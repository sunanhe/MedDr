DATASET="PCam200"
ARCH="vit_base_patch16_224"

CUBLAS_WORKSPACE_CONFIG=:16:8 python inference_specialist.py \
    --config_file 'src/config/config.yaml' \
    --dataset ${DATASET} \
    --img_size 224 \
    --training_procedure 'endToEnd' \
    --architecture ${ARCH} \
    --seed 9930641