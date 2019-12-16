import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle 
import argparse 
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np 

# training score will be in filename 
# unchunked:  1000 x 100 x 3 x 64 x 64  ->  
# chunked : 1000 x 10 x 3 x 10 x 64 x 64 
# data : 1000 x 10 

class Model(nn.Module): 
    def __init__(self, img_model, seq_model):
        super().__init__() 

        self.img_model, self.seq_model = None, None

        if img_model == "slow_fusion":
            from models.slow_fusion import SlowFusion 
            self.img_model = SlowFusion(3, 10, 64)
        elif img_model == "early_fusion": 
            from models.early_fusion import EarlyFusion
            self.img_model = EarlyFusion(3, 10, 64)
        elif img_model == "late_fusion": 
            from models.late_fusion import LateFusion
            self.img_model = LateFusion(3, 10, 64)
        elif img_model == "vanilla_cnn":
            from models.basic_cnn import BasicCNN
            self.img_model = BasicCNN(3, 64)
        else: 
            from models.imagenet_model_wrapper import ImageNet_Model_Wrapper
            self.img_model = ImageNet_Model_Wrapper(img_model)

        if seq_model == "vanilla_rnn": 
            from models.rnn import RNN
            self.seq_model = RNN(512, 256, 2)
        elif seq_model == "lstm": 
            from models.lstm import LSTM
            self.seq_model = LSTM(512, 256, num_layers=2, dropout=0.1, bidirectional=True)
        elif seq_model == "lstmn": 
            from models.lstmn import BiLSTMN
            self.seq_model = BiLSTMN(512, 256, num_layers=2, dropout=0.1, tape_depth=10)
        elif seq_model == "transformer_abs": 
            from models.transformer import Transformer 
            self.seq_model = Transformer(512, 8)
        elif seq_model == "transformer_rel": 
            raise ValueError("not yet implemented")

        # attention over seq_model output
        self.query_vector = nn.Parameter(torch.randn(1, 64))
        # self.attn_w  = nn.Bilinear(64, 512, 1)
        self.attn_w = nn.Parameter(torch.randn(64, 512))

        self.linear1 = nn.Linear(512, 32)
        self.linear2 = nn.Linear(32, 1)
        
    def forward(self, x): 
        # run cnn: img_data -> 512
        embed = self.img_model(x)
        # print(f"embed_post_img: {embed.size()}")
        # output: (100, 512)

        # unsqueeze to (100, 1, 512)
        embed = embed.unsqueeze(1)

        embed = self.seq_model(embed)
        # print(embed.shape)
        # output: (frame, 512)

        attn = torch.mm(torch.mm(self.query_vector, self.attn_w), embed.t())
        # output: (1, frame)
        attn = F.softmax(attn, dim=1).t()

        ctxt = torch.sum(attn * embed, dim=0)

        embed = F.relu(self.linear1(ctxt)) 
        return torch.sigmoid(self.linear2(embed))



def train(epochs, model, train_iter, eval_iter, model_name, device, tolerance=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_losses = []
    eval_losses = []
    strikes = 0
    best_eval_loss = float('inf')

    for epoch in tqdm(range(1, epochs + 1)):
        print(f"epoch: {epoch}")
        train_loss = 0
        model.train()
        print("TRAINING")
        train_count = 0
        for ex, rt_score in train_iter:
            train_count += 1 
            optimizer.zero_grad()
            out = model(ex.to(device))
            loss = criterion(out.squeeze(), rt_score.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss/train_count)
        print(f"training loss: {train_loss}")

        print("EVALUATING")
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
        print(f"eval loss: {eval_loss}")
        
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), f"./_trained_models/{model_name}_checkpoint.pt")
            pickle.dump(train_losses, open(f"./_trained_models/{model_name}_train_losses.p", "wb"))
            pickle.dump(eval_losses, open(f"./_trained_models/{model_name}_eval_losses.p", "wb"))
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

# for sm in ["vanilla_rnn", "lstm", "lstmn", "transformer_abs"]: 
#     for im in ['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn']: 
#         print(f"TRYING EXAMPLE: {sm}, {im}")
#         model = Model(im, sm)
#         chunked_needed = im in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
#         data = None 
#         # labels = pickle.load(open("./data/labels.p", "rb"))
#         if chunked_needed:
#             # data = pickle.load(open("./data/chunked_data.p", "rb"))
#             data = torch.randn(5, 10, 3, 10, 64, 64)
#         else: 
#             # data = pickle.load(open("./data/data.p", "rb"))
#             data = torch.randn(5, 100, 3, 64, 64)
#         for peh in data: 
#             print(model(peh))
#             break


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_model", type=str, help="name of time series model", required=True, 
                        choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "transformer_abs"])
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
        data = pickle.load(open("./_data/chunked_data.p", "rb"))
        # data = torch.randn(5, 10, 3, 10, 64, 64)
    else: 
        data = pickle.load(open("./_data/data.p", "rb"))
        # data = torch.randn(5, 100, 3, 64, 64)
    labels = pickle.load(open("./_data/labels.p", "rb"))
    # labels = torch.zeros(data.shape[0])
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2)
    X_train, X_test, Y_train, Y_test = list(map(lambda x : torch.from_numpy(x).float(), (X_train, X_test, Y_train, Y_test)))
    train_iter = list(zip(X_train, Y_train))
    eval_iter = list(zip(X_test, Y_test))

    print("Creating Model....")
    model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"
    model = Model(args.img_model, args.seq_model).to(device)
    
    print("Training Model....")
    train(5000, model, train_iter, eval_iter, model_name, device)
    print(f"{model_name} training complete")

if __name__ == "__main__" : main()