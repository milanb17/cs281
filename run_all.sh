# !/bin/bash 

echo "fixing SEQ as lstm"

for fast_img in vanilla_cnn slow_fusion early_fusion late_fusion; do 
    echo "working on timing_SEQ_lstm_IMG_${fast_img}..."
    { time python train.py --seq_model lstm --img_model $fast_img --gpu 0; } 2> "./trained_models_new/timing_SEQ_lstm_IMG_${fast_img}.txt"
done 

echo "fixing IMG MODEL as slow_fusion"

for fast_seq in vanilla_rnn lstm lstmn transformer_abs; do 
    echo "working on timing_SEQ_${fast_seq}_IMG_slow_fusion..."
    { time python train.py --seq_model $fast_seq --img_model slow_fusion --gpu 0; } 2> "./trained_models_new/timing_SEQ_${fast_seq}_IMG_slow_fusion.txt"
done 