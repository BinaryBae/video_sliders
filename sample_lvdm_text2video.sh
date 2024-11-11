PROMPT="human eating banana"
OUTDIR="results/t2v"

CKPT_PATH="t2v.ckpt"
CONFIG_PATH="text2video.yaml"

python sample_text2video.py \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    --save_jpg
