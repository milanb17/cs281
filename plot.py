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
 
beautify = { "vanilla_rnn" : "RNN"
            , "lstm" : "LSTM"
            , "lstmn" : "LSTMN"
            , "transformer_abs" : "Transformer"
            , "early_fusion" : "Early Fusion"
            , "late_fusion" : "Late Fusion"
            , "slow_fusion" : "Slow Fusion"
            , "stack_lstm" : "Stack LSTM"
            , "resnet" : "ResNet (ImageNet weights)"
            , "densenet" : "DenseNet (ImageNet weights)"
            , "vgg" : "VGG16 (ImageNet weights)"
            , "vanilla_cnn" : "CNN"
            , "g_train_losses" : "Granular Loss (MSE)"
            , "train_losses" : "Training Loss (MSE)"
            , "eval_losses" : "Evaluation Loss (MSE)"
            }

def plot_histogram():
    scores = pickle.load(open("./scores.p", "rb")) * 100
    ax = sns.distplot(scores)
    fig = ax.get_figure()
    plt.xlabel("Movie Ratings")
    plt.ylabel("Frequency")
    # plt.xlim(0, 100)
    plt.title("Score Histogram w/ KDE")
    fig.savefig("./plots/score_hist.png")



def plot_mse(data, seq_model_name, img_model_name, graph_type, show=True, save=True): 
    x_axis_data = np.arange(len(data)) + 1 
    plt.plot(x_axis_data, data, "-")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    if show: 
        plt.title(f"{beautify[seq_model_name]} x {beautify[img_model_name]} on {beautify[graph_type]}")
    else: 
        plt.title(f"{beautify[seq_model_name]} x {beautify[img_model_name]}")
    if save:
        plt.savefig(f"./plots/SEQ_{seq_model_name}_IMG_{img_model_name}__{graph_type}.png")
    if show:
        plt.show()


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-pp", "--paper_plots", action="store_true", default=False)
    parser.add_argument("--which", choices=["eval", "g_train", "train"], required=False)
    parser.add_argument("-plot_score_hist", "--plot_score_hist", action="store_true", default=False)
    parser.add_argument("-plot_train", "--plot_train", action="store_true", default=False)
    parser.add_argument("-plot_eval", "--plot_eval", action="store_true", default=False)
    parser.add_argument("-plot_g_train", "--plot_g_train", action="store_true", default=False)
    parser.add_argument("-pa", "--plot_all", action="store_true", default=False)
    parser.add_argument("--seq_model", type=str, help="name of time series model", required=False, 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "transformer_abs", "stack_lstm"])
    parser.add_argument("--img_model", type=str, help="name of img processing model name", required=False, 
                        choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
    args = parser.parse_args()

    if args.paper_plots:
        assert(args.which is not None and args.which != "")
        plt.rcParams.update({"font.size" : 8})
        seq_model_fixed = "lstm"
        img_models_vary = ["vanilla_cnn", "late_fusion",  "early_fusion", "slow_fusion"]
        name = f"{args.which}_losses"

        plt.figure().suptitle(f"LSTM x Image Model, Increasing Time Sensitive Words; {beautify[name]}", fontsize=15)

        for i, img_model in enumerate(img_models_vary):
            model_name = f"SEQ_{seq_model_fixed}_IMG_{img_model}"
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_{args.which}_losses.p", "rb"))
            plt.subplot(2, len(img_models_vary) // 2, i+1)
            plot_mse(data, seq_model_fixed, img_model, name, show=False, save=False)
        plt.show()

        # seq_model_fixed = "lstm"
        # img_models_vary = ["resnet", "densenet", "vgg"]
        # name = f"{args.which}_losses"

        # plt.figure().suptitle(f"LSTM x Image Model, ImageNet; {beautify[name]}", fontsize=15)

        # for i, img_model in enumerate(img_models_vary):
        #     model_name = f"SEQ_{seq_model_fixed}_IMG_{img_model}"
        #     data = pickle.load(open(f"./trained_models_new/results/{model_name}_{args.which}_losses.p", "rb"))
        #     plt.subplot(1, len(img_models_vary), i+1)
        #     plot_mse(data, seq_model_fixed, img_model, name, show=False, save=False)
        # plt.show()

        img_model_fixed = "slow_fusion"
        seq_model_vary = ["vanilla_rnn", "lstm", "lstmn", "transformer_abs"]

        plt.figure().suptitle(f"Seq Models x Slow Fusion, Increasing Model Complexity; {beautify[name]}", fontsize=15)
        for i, seq_model in enumerate(seq_model_vary):
            model_name = f"SEQ_{seq_model}_IMG_{img_model_fixed}"
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_{args.which}_losses.p", "rb"))
            plt.subplot(2, len(seq_model_vary) // 2, i+1)
            plot_mse(data, seq_model, img_model_fixed, name, show=False, save=False)
        plt.show()

        # curiosities 

        models = [("stack_lstm", "slow_fusion"), ("lstm", "vgg")]
        plt.figure().suptitle(f"Other Interesting Models; {beautify[name]}", fontsize=15)
        for i, (seq, img) in enumerate(models): 
            model_name = f"SEQ_{seq}_IMG_{img}"
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_{args.which}_losses.p", "rb"))
            plt.subplot(1, len(models), i+1)
            plot_mse(data, seq, img, name, show=False, save=False)
        plt.show()


    elif args.plot_score_hist: 
        plot_histogram()
    else:
        assert(args.seq_model)
        assert(args.img_model)
        model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
        if args.plot_all: 
            args.plot_eval = args.plot_g_train = args.plot_train = True 
        if args.plot_g_train:
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_g_train_losses.p", "rb"))
            plot_mse(data, args.seq_model, args.img_model, "g_train_losses")
        if args.plot_train: 
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_train_losses.p", "rb"))
            plot_mse(data, args.seq_model, args.img_model, "train_losses")
        if args.plot_eval: 
            data = pickle.load(open(f"./trained_models_new/results/{model_name}_eval_losses.p", "rb"))[:16]
            plot_mse(data, args.seq_model, args.img_model, "eval_losses")


if __name__ == "__main__" : main()

