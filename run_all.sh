# !/bin/bash 

for fast_seq in vanilla_rnn lstm; do 
    for fast_img in early_fusion slow_fusion vanilla_cnn; do 
        # echo "${fast_seq} ${fast_img}"
        python train.py --seq_model $fast_seq --img_model $fast_img --gpu 0
    done 
done 

echo "done with fast, starting slow" 

for slow_seq in lstmn transformer_abs; do 
    for slow_img in late_fusion resnet densenet vgg; do 
        # echo "${slow_seq} ${slow_img}"
        python train.py --seq_model $slow_seq --img_model $slow_img --gpu 0
    done 
done 

echo "done"