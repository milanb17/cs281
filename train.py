import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle 
import argparse 

# training score will be in filename 
# unchunked:  1000 x 100 x 3 x 64 x 64  ->  
# chunked : 1000 x 10 x 3 x 10 x 64 x 64 
# data : 1000 x 10 

model = None
train_iter = None
eval_iter = None


parser = argparse.ArgumentParser()
parser.add_argument("--seq_model", type=str, help="name of time series model", required=True, 
                    choices=["vanilla_rnn", "lstm", "lstmn", "transformer_rel", "transformer_abs"])
parser.add_argument("--img_model", type=str, help="name of img processing model name", required=True, 
                    choices=['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn'])
args = parser.parse_args()

model_name = f"SEQ_{args.seq_model}_IMG_{args.img_model}"

chunked_needed = args.img_model in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
data = None 
# labels = pickle.load(open("./data/labels.p", "rb"))
if chunked_needed:
    # data = pickle.load(open("./data/chunked_data.p", "rb"))
    data = torch.randn(5, 10, 3, 10, 64, 64)
else: 
    # data = pickle.load(open("./data/data.p", "rb"))
    data = torch.randn(5, 100, 3, 64, 64)

class Model(nn.Module): 
    def __init__(self):
        super().__init__() 

        self.img_model, self.seq_model = None, None

        if args.img_model == "slow_fusion":
            from models.slow_fusion import SlowFusion 
            self.img_model = SlowFusion(3, 10, 64)
        elif args.img_model == "early_fusion": 
            from models.early_fusion import EarlyFusion
            self.img_model = EarlyFusion(3, 10, 64)
        elif args.img_model == "late_fusion": 
            from models.late_fusion import LateFusion
            self.img_model = LateFusion(3, 10, 64)
        elif args.img_model == "vanilla_cnn":
            from models.basic_cnn import BasicCNN
            self.img_model = BasicCNN(3, 64)
        else: 
            from models.imagenet_model_wrapper import ImageNet_Model_Wrapper
            self.img_model = ImageNet_Model_Wrapper(args.img_model)

        if args.seq_model == "vanilla_rnn": 
            from models.rnn import RNN
            self.seq_model = RNN(512, 256, 2)
        elif args.seq_model == "lstm": 
            from models.lstm import LSTM
            self.seq_model = LSTM(512, 256, num_layers=2, dropout=0.1, bidirectional=True)
        elif args.seq_model == "lstmn": 
            from models.lstmn import BiLSTMN
            self.seq_model = BiLSTMN(512, 256, num_layers=2, dropout=0.1, tape_depth=10)
        elif args.seq_model == "transformer_abs": 
            from models.transformer import Transformer 
            self.seq_model = Transformer(512, 8)

        self.linear = nn.Linear(512, 1)
        
    def forward(self, x): 
        # run cnn: img_data -> 512
        embed = self.img_model(x)
        print(f"embed_post_img: {embed.size()}")
        # output: (100, 512)

        # unsqueeze to (100, 1, 512)
        embed = embed.unsqueeze(1)

        embed = self.seq_model(embed)
        # output: (512)
        print(f"embed_post_seq: {embed.size()}")

        embed = self.linear(embed)
        print(f"embed_post_linear: {embed.size()}")
        return embed 


model = Model()
for peh in data:
    model(peh)




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

