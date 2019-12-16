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

def plot_g_train(model_name): 
    data = pickle.load(open(f"./trained_models/{model_name}_g_train_losses.p", "rb"))[:15]
    x_axis_data = np.arange(len(data))
    plt.plot(x_axis_data, data, "-")
    plt.show()

def plot_train(model_name): 
    data = pickle.load(open(f"./trained_models/{model_name}_train_losses.p", "rb"))
    print(data)
    x_axis_data = np.arange(len(data))
    plt.plot(x_axis_data, data, "-")
    plt.show()

def plot_eval(model_name): 
    data = pickle.load(open(f"./trained_models/{model_name}_eval_losses.p", "rb"))
    x_axis_data = np.arange(len(data))
    plt.plot(x_axis_data, data, "-")
    plt.show()


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-plot_score_hist", "--plot_score_hist", action="store_true", default=False)
    parser.add_argument("-plot_train", "--plot_train", action="store_true", default=False)
    parser.add_argument("-plot_eval", "--plot_eval", action="store_true", default=False)
    parser.add_argument("-plot_g_train", "--plot_g_train", action="store_true", default=False)
    parser.add_argument("--seq_model", type=str, help="name of time series model", 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "transformer_abs"])
    parser.add_argument("--img_model", type=str, help="name of img processing model name", 
                        choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
    args = parser.parse_args()

    if args.plot_score_hist: 
        plot_histogram()
    if args.plot_g_train: 
        model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
        plot_g_train(model_name)
    if args.plot_train: 
        model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
        plot_train(model_name)
    if args.plot_eval: 
        model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
        plot_eval(model_name)


if __name__ == "__main__" : main()

