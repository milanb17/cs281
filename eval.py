import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle 
import argparse 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np 
import itertools

# Take available models, evaluate RMSE on random sample of dataset 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_model", type=str, help="name of time series model", required=True, 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "stack_lstm"])
    parser.add_argument("--img_model", type=str, help="name of img processing model name", required=True, 
                        choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
    parser.add_argument("--gpu", type=int, help="which gpu to run on", required=True, choices=[0, 1])
    args = parser.parse_args()

    hold_seq_fixed = "lstm"
    cnn_options = ['early_fusion', 'late_fusion', 'slow_fusion', 'vanilla_cnn']
    


if __name__ == "__main__" : main()


