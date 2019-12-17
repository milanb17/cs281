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

# training score will be in filename 
# unchunked:  1000 x 100 x 3 x 64 x 64  ->  
# chunked : 1000 x 10 x 3 x 10 x 64 x 64 
# data : 1000 x 10 


def train(epochs, model, train_iter, eval_iter, model_name, device, tolerance=5, bound=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_losses = []
    eval_losses = []
    g_train_losses = []
    strikes = 0
    best_eval_loss = float('inf')

    for epoch in range(epochs + 1):
        pickle.dump(train_losses, open(f"./trained_models_new/{model_name}_train_losses.p", "wb"))
        pickle.dump(eval_losses, open(f"./trained_models_new/{model_name}_eval_losses.p", "wb"))
        pickle.dump(g_train_losses, open(f"./trained_models_new/{model_name}_g_train_losses.p", "wb"))
        if epoch == 0:
            with torch.no_grad():
                print(f"epoch: {epoch} -- initial scores")
                train_loss = 0
                model.eval()
                train_count = 0
                for ex, rt_score in train_iter:
                    train_count += 1 
                    out = model(ex.to(device))
                    loss = criterion(out.squeeze(), rt_score.to(device))
                    train_loss += loss.item()
                print(f"training loss: {train_loss/train_count}")
                train_losses.append(train_loss / train_count)

                eval_loss = 0
                eval_count = 0
                for ex, rt_score in eval_iter:
                    out = model(ex.to(device))
                    loss = criterion(out.squeeze(), rt_score.to(device))
                    eval_loss += loss.item()
                    eval_count += 1 
                print(f"eval loss: {eval_loss/eval_count}")
                eval_losses.append(eval_loss/eval_count)
            continue
            
        print(f"epoch: {epoch}")
        train_loss = 0
        g_train_loss = 0
        model.train()
        train_count = 0
        for ex, rt_score in train_iter:
            train_count += 1 
            optimizer.zero_grad()
            out = model(ex.to(device))
            loss = criterion(out.squeeze(), rt_score.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            g_train_loss += loss.item()
            if train_count % 50 == 0:
                g_train_losses.append(g_train_loss/50)
                g_train_loss = 0

        train_losses.append(train_loss/train_count)
        print(f"training loss: {train_loss/train_count}")
        eval_loss = 0
        model.eval()
        with torch.no_grad():
            eval_count = 0
            for ex, rt_score in eval_iter:
                out = model(ex.to(device))
                loss = criterion(out.squeeze(), rt_score.to(device))
                eval_loss += loss.item()
                eval_count += 1 
            eval_losses.append(eval_loss/eval_count)
            scheduler.step(eval_losses[-1])
        print(f"eval loss: {eval_loss/eval_count}")
        
        if best_eval_loss - eval_loss > bound:
            torch.save(model.state_dict(), f"./trained_models_new/{model_name}_checkpoint.pt")
            best_eval_loss = eval_loss
            strikes = 0
        elif strikes == tolerance:
            print("Early Stopping")
            break
        else:
            strikes += 1



# model = Model(args.img_model, args.seq_model)
# for peh in data:
#     print(model(peh))

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_model", type=str, help="name of time series model", required=True, 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_abs", "stack_lstm"])
    parser.add_argument("--img_model", type=str, help="name of img processing model name", required=True, 
                        choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
    parser.add_argument("--gpu", type=int, help="which gpu to run on", required=True, choices=[0, 1])
    args = parser.parse_args()
    chunked_needed = args.img_model in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # import sys 
    # print(f"Args for SEQ:{args.seq_model}, IMG:{args.img_model} recieved correctly")
    # sys.exit(0)
    data = None 
    print("Getting Data...")
    if chunked_needed:
        data = pickle.load(open("./converter/trailers_chunked.p", "rb"))
        # data = torch.randn(5, 10, 3, 10, 64, 64)
    else: 
        data = pickle.load(open("./converter/trailers_normal.p", "rb"))
        # data = torch.randn(5, 100, 3, 64, 64)
    labels = pickle.load(open("./converter/scores.p", "rb"))
    # labels = torch.zeros(data.shape[0])
    # print("Data Loaded")
    # print(labels.shape)
    # print(data.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15)
    # print("Train Test Split Generated")
    X_train, X_test, Y_train, Y_test = list(map(lambda x : torch.from_numpy(x).float(), (X_train, X_test, Y_train, Y_test)))
    # print("Tensors Generated")
    train_iter = list(zip(X_train, Y_train))
    eval_iter = list(zip(X_test, Y_test))
    # print("Training/Eval Iterators Generated")

    # print("Creating Model....")
    model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
    model = Model(args.img_model, args.seq_model).to(device)
    
    print("Training Model....")
    train(500, model, train_iter, eval_iter, model_name, device, tolerance=15)
    print(f"{model_name} training complete")

if __name__ == "__main__" : main()