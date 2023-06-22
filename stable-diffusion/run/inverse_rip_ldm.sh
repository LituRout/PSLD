export CUDA_VISIBLE_DEVICES='1'
python scripts/inverse.py \
    --file_id='00014.png' \
    --task_config='configs/inpainting_config.yaml' \
    --inpainting=1 \
    --general_inverse=0 \
    --gamma=1e-1 \
    --omega=1e-1 \
    --ffhq256 \
    --W=256 \
    --H=256 \
    --C=3 \
    --f=4 \
    --outdir='outputs/psld-ldm-samples-rip'