# python run_evals.py --eval_imgs True --eval_bits False  --img_dir /data/haoyuanliu/project/cat.png --img_dir_nw path/to/imgs_nw 
python run_evals.py --eval_imgs False --eval_bits True  --img_dir /data/haoyuanliu/project/stable_signature/wm_images/wm/ \
    --key_str '111010110101000001010111010011010100010000100111' \
    --msg_decoder_path /data/haoyuanliu/project/stable_signature/models/other_dec_48b_whit.torchscript.pt \