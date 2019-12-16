# !/bin/bash 

echo "starting..."

for fast_seq in vanilla_rnn lstm lstmn transformer_abs; do 
    for fast_img in vanilla_cnn slow_fusion early_fusion late_fusion; do 
        echo "working on timing_SEQ_${fast_seq}_IMG_${fast_img}..."
        { time python train.py --seq_model $fast_seq --img_model $fast_img --gpu 0; } 2> "./trained_models_new/timing_SEQ_${fast_seq}_IMG_${fast_img}.txt"
    done 
done 

echo "done with fast, starting imagenet models..." 

for slow_seq in vanilla_rnn lstm lstmn transformer_abs; do 
    for slow_img in resnet densenet vgg; do 
        echo "working on timing_SEQ_${slow_seq}_IMG_${slow_img}..."
        { time python train.py --seq_model $slow_seq --img_model $slow_img --gpu 0; } 2> "./trained_models_new/timing_SEQ_${slow_seq}_IMG_${slow_img}.txt"
    done 
done 

echo "done"