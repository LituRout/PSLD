export CUDA_VISIBLE_DEVICES='1'
python scripts/inverse.py \
    --file_id='00014.png' \
    --task_config='configs/inpainting_config_psld.yaml' \
    --inpainting=1 \
    --general_inverse=0 \
    --outdir='outputs/psld-samples-rip';