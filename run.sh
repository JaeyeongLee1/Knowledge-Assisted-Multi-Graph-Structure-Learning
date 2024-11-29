gpu_n=$1
dim=128
alp=0.3
topk=10
out_mode=1
out_num=2
if [[ "$gpu_n" == "cpu" ]]; then
    python main.py\
        -batch 64\
        -emb_dim $dim\
        -feature_dim $dim\
        -alpha $alp\
        -topk $topk\
        -dataset 'wadi'\
        -early_stop_win 5\
        -out_mode $out_mode\
        -out_layer_num $out_num\
        -scale_bool
else
    CUDA_VISIBLE_DEVICES=$gpu_n python main.py\
        -batch 64\
        -emb_dim $dim\
        -feature_dim $dim\
        -alpha $alp\
        -topk $topk\
        -dataset 'wadi'\
        -early_stop_win 5\
        -out_mode $out_mode\
        -out_layer_num $out_num\
        -scale_bool