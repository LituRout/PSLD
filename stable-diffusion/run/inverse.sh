export CUDA_VISIBLE_DEVICES='0'
python scripts/inverse.py \
    --file_id='00015.png' \
    --task_config='configs/super_resolution_config_psld.yaml' \
    --outdir='outputs/psld-samples-sr';