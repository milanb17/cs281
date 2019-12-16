# !/bin/bash 

for fast_seq in vanilla_rnn lstm; do 
    for fast_img in early_fusion slow_fusion vanilla_cnn; do 
        (time python train.py --seq_model $fast_seq --img_model $fast_img --gpu 0) &> "./trained_models/timing_SEQ_${fast_seq}_IMG_${fast_img}.txt"
    done 
done 

echo "done with fast, starting slow" 

for slow_seq in lstmn transformer_abs; do 
    for slow_img in late_fusion resnet densenet vgg; do 
        # echo "${slow_seq} ${slow_img}"
        (time python train.py --seq_model $slow_seq --img_model $slow_img --gpu 0) &> "./trained_models/timing_SEQ_${slow_seq}_IMG_${slow_img}.txt"
    done 
done 

echo "done"