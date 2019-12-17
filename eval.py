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
from base_model import Model

# Take available models, evaluate RMSE on random sample of dataset 
# Slow_fusion x [vanilla_rnn lstm lstmn transformer_abs]
def eval(model, data, device):
    criterion = nn.MSELoss()
    rmse = 0
    i = 0
    model.eval()
    with torch.no_grad():
        for ex, rt_score in data:
            i += 1
            out = model(ex.to(device))
            rmse_i = criterion(out.squeeze(), rt_score.to(device))
            rmse += rmse_i.item()
    
    return (rmse/i) ** 0.5

def eval_model(img_net, seq_net, device, data):
    model = Model(img_net, seq_net).to(device)
    model_name = f"SEQ_{img_net}_IMG_{seq_net}"
    model_path = f"./trained_models_new/{model_name}_checkpoint.pt"
    print("LOADING MODEL")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("MODEL LOADED - EVAL STARTING")
    return eval(model, data, device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)

    chunked_data = pickle.load(open("./converter/trailers_chunked.p", "rb"))
    data = pickle.load(open("./converter/trailers_normal.p", "rb")) 
    labels = pickle.load(open("./converter/scores.p", "rb"))
    chunked_data, data, labels = list(map(lambda x: torch.from_numpy(x).float(), (chunked_data, data, labels)))

    chunked_data = list(zip(chunked_data, labels))
    data = list(zip(data, labels))

    hold_seq_fixed = "lstm"
    cnn_options = ['early_fusion', 'late_fusion', 'slow_fusion', 'vanilla_cnn']

    results = []

    for cnn in cnn_options:
        chunked_needed = cnn in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
        res = eval_model(cnn, hold_seq_fixed, device, chunked_data if chunked_needed else data)
        results.append(((cnn, hold_seq_fixed), res))
    
    cnn_f = "slow_fusion"
    seq_v = ["vanilla_rnn", "lstm", "lstmn", "transformer_abs", "lstm_stack"]
    for seq in seq_v:
        chunked_needed = cnn_f in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
        res = eval_model(cnn_f, seq, device, chunked_data if chunked_needed else data)
        results.append(((cnn_f, seq), res))

    pickle.dump(results, open(f"./results.p", "wb"))

if __name__ == "__main__" : main()
