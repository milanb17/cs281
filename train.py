import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle

model = None
train_iter = None
eval_iter = None

def train(epochs, tolerance=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train_losses = []
    eval_losses = []
    strikes = 0
    best_eval_loss = float('inf')

    for epoch in tqdm(range(1, epochs + 1)):
        print(f"epoch: {epoch}")
        train_loss = 0
        model.train()
        print("TRAINING")
        for ex, rt_score in train_iter:
            optimizer.zero_grad()
            out = model(ex)
            loss = criterion(out, rt_score)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss)
        print(f"training loss: {train_loss}")

        print("EVALUATING")
        eval_loss = 0
        model.eval()
        with torch.no_grad():
            for ex, rt_score in eval_iter:
                out = model(ex)
                loss = criterion(out, rt_score)
                eval_loss += loss.item()
            eval_losses.append(eval_loss)
        print(f"eval loss: {eval_loss}")
        
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), f"{model_name}_checkpoint.pt")
            pickle.dump(train_losses, open(f"{model_name}_train_losses.p", "wb"))
            pickle.dump(eval_losses, open(f"{model_name}_eval_losses.p", "wb"))
            best_eval_loss = eval_loss
            strikes = 0
        elif strikes == tolerance:
            print("Early Stopping")
            break
        else:
            strikes += 1

