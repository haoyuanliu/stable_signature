unset LD_LIBRARY_PATH
python finetune_ldm_decoder.py --num_keys 1 \
    --ldm_config /data/haoyuanliu/project/stable_signature/v1-inference.yaml \
    --ldm_ckpt /data/haoyuanliu/project/huggingface/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt \
    --msg_decoder_path /data/haoyuanliu/project/stable_signature/models/other_dec_48b_whit.torchscript.pt \
    --train_dir /home/haoyuanliu/project/AquaLoRA/coco2017/test2017/ \
    --val_dir /home/haoyuanliu/project/AquaLoRA/coco2017/val2017/ \
    --steps 1000 \
    --batch_size 5 \