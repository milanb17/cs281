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

        # attention over seq_model output
        self.query_vector = nn.Parameter(torch.randn(64))
        self.attn_wh = nn.Bilinear(64, 512, 32)
        self.attn_wx = nn.Linear(32, 1)

        self.linear = nn.Linear(512, 1)
        
    def forward(self, x): 
        # run cnn: img_data -> 512
        embed = self.img_model(x)
        # print(f"embed_post_img: {embed.size()}")
        # output: (100, 512)

        # unsqueeze to (100, 1, 512)
        embed = embed.unsqueeze(1)

        embed = self.seq_model(embed)
<<<<<<< HEAD
        # output: (frame, 512)

        attn = self.attn_wh(self.query_vector, embed)
        # output: (frame, 32)

        attn = self.attn_wx(attn)
        # output: (frame, 1)
        
        attn = torch.softmax(attn)

        ctxt = torch.sum(attn * embed)
        # output: (1, 512)

        embed = self.linear(ctxt)
        print(f"embed_post_linear: {embed.size()}")
=======
        # output: (512)
        # print(f"embed_post_seq: {embed.size()}")

        embed = torch.sigmoid(self.linear(embed))
        # print(f"embed_post_linear: {embed.size()}")
>>>>>>> 92111ba89faa0555c76601f9f187645fbb515905
        return embed 


# model = Model(args.img_model, args.seq_model)
# for peh in data:
#     print(model(peh))

for sm in ["vanilla_rnn", "lstm", "lstmn", "transformer_abs"]: 
    for im in ['early_fusion', 'late_fusion', 'slow_fusion', 'resnet', 'densenet', 'vgg', 'vanilla_cnn']: 
        print(f"TRYING EXAMPLE: {sm}, {im}")
        model = Model(im, sm)
        chunked_needed = im in frozenset(["slow_fusion", "early_fusion", "late_fusion"])
        data = None 
        # labels = pickle.load(open("./data/labels.p", "rb"))
        if chunked_needed:
            # data = pickle.load(open("./data/chunked_data.p", "rb"))
            data = torch.randn(5, 10, 3, 10, 64, 64)
        else: 
            # data = pickle.load(open("./data/data.p", "rb"))
            data = torch.randn(5, 100, 3, 64, 64)
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

