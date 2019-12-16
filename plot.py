import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle 
import argparse 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns  

def plot_histogram():
    scores = pickle.load(open("./scores.p", "rb")) * 100
    ax = sns.distplot(scores)
    fig = ax.get_figure()
    plt.xlabel("Movie Ratings")
    plt.ylabel("Frequency")
    # plt.xlim(0, 100)
    plt.title("Score Histogram w/ KDE")
    fig.savefig("./plots/score_hist.png")



def plot_mse(data, seq_model_name, img_model_name, graph_type): 
    beautify = { "vanilla_rnn" : "RNN"
                , "lstm" : "LSTM"
                , "lstmn" : "LSTMN","transformer_abs" : "Transformer"
                , "early_fusion" : "Early Fusion"
                , "late_fusion" : "Late Fusion"
                , "slow_fusion" : "Slow Fusion"
                , "resnet" : "ResNet (ImageNet weights)"
                , "densenet" : "DenseNet (ImageNet weights)"
                , "vgg" : "VGG16 (ImageNet weights)"
                , "vanilla_cnn" : "CNN"
                , "g_train_losses" : "Granular Loss (RMSE)"
                , "train_losses" : "Training Loss (RMSE)"
                , "eval_losses" : "Evaluation Loss (RMSE)"
                }
    x_axis_data = np.arange(len(data)) + 1 
    plt.plot(x_axis_data, data, "-")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"MSE of {beautify[seq_model_name]} Seq Model x {beautify[img_model_name]} Img Model on {beautify[graph_type]}")
    plt.show()
    plt.savefig(f"./plots/SEQ_{seq_model_name}_IMG_{img_model_name}__{graph_type}.png")


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-plot_score_hist", "--plot_score_hist", action="store_true", default=False)
    parser.add_argument("-plot_train", "--plot_train", action="store_true", default=False)
    parser.add_argument("-plot_eval", "--plot_eval", action="store_true", default=False)
    parser.add_argument("-plot_g_train", "--plot_g_train", action="store_true", default=False)
    parser.add_argument("-plot_all_losses", "--plot_all_losses", action="store_true", default=False)
    parser.add_argument("--seq_model", type=str, help="name of time series model", required=True, 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "transformer_abs"])
    parser.add_argument("--img_model", type=str, help="name of img processing model name", required=True, 
                        choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
    args = parser.parse_args()

    if args.plot_score_hist: 
        plot_histogram()
    model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
    if args.plot_all_losses: 
        args.plot_eval = args.plot_g_train = args.plot_train = True 
    if args.plot_g_train:
        data = pickle.load(open(f"./trained_models/{model_name}_g_train_losses.p", "rb"))
        plot_mse(data, args.seq_model, args.img_model, "g_train_losses")
    if args.plot_train: 
        data = pickle.load(open(f"./trained_models/{model_name}_train_losses.p", "rb"))
        plot_mse(data, args.seq_model, args.img_model, "train_losses")
    if args.plot_eval: 
        data = pickle.load(open(f"./trained_models/{model_name}_eval_losses.p", "rb"))
        plot_mse(data, args.seq_model, args.img_model, "eval_losses")


if __name__ == "__main__" : main()

