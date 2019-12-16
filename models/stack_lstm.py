import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMStack(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(LSTMStack, self).__init__()
        self.rnn = nn.LSTMCell(embedding_dim + hidden_size, hidden_size)
        self.d_layer = nn.Linear(hidden_size, 1)
        self.u_layer = nn.Linear(hidden_size, 1)
        self.v_layer = nn.Linear(hidden_size, hidden_size)
        self.o_layer = nn.Linear(hidden_size, hidden_size)
        self.rnn_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, inpt, state):
        prev_hidden, prev_r, prev_V, prev_s = state
        rnn_input = torch.cat((inpt, prev_r), dim=1)
        rnn_output, new_hidden = self.rnn(rnn_input, prev_hidden)
        rnn_output = self.rnn_out(rnn_output)
        d_t = self.d_layer(rnn_output)
        u_t = self.u_layer(rnn_output)
        v_t = self.v_layer(rnn_output)
        o_t = self.o_layer(rnn_output)

        if prev_V is None:
            new_V = v_t.detach()
        else:
            new_V = torch.cat((prev_V, v_t.detach()), dim=0)

        if prev_s is None:
            new_s = d_t.detach()
        else:
            shid_prev_s = torch.flip(torch.cumsum(torch.flip(prev_s, [0]), dim=0), [0]) - prev_s
            new_s = torch.clamp(prev_s - torch.clamp(u_t.item() - shid_prev_s, min=0), min=0)
            new_s = torch.cat((new_s, d_t.detach()), dim=0)

        shid_new_s = torch.flip(torch.cumsum(torch.flip(new_s, [0]), dim=0), [0]) - new_s
        r_scalars = torch.min(new_s, torch.clamp(1 - shid_new_s, min=0))
        new_r = torch.sum(r_scalars.view(-1, 1) * new_V, dim=0).view(1, -1)
        return o_t, ((rnn_output, new_hidden), new_r, new_V, new_s)

class EncoderLSTMStack(nn.Module):
    def __init__(self, embedding_dim, hidden_size):
        super(EncoderLSTMStack, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_f = LSTMStack(embedding_dim, hidden_size)
        self.rnn_b = LSTMStack(embedding_dim, hidden_size)
    def forward(self, embeds):
        x = torch.zeros(1, self.hidden_size, device=device)
        state_f = ((x.clone(), x.clone()), x.clone(), None, None)
        state_b = ((x.clone(), x.clone()), x.clone(), None, None)
        mem_f = torch.zeros(embeds.size(0), self.hidden_size, device=device)
        mem_b = torch.zeros(embeds.size(0), self.hidden_size, device=device)

        for i in range(embeds.size(0)):
            embeds_input = embeds[i].view(1, -1)
            o_t, state_f = self.rnn_f(embeds_input, state_f)
            mem_f[i] = o_t
        for i in reversed(range(embeds.size(0))):
            embeds_input = embeds[i].view(1, -1)
            o_t, state_b = self.rnn_b(embeds_input, state_b)
            mem_b[i] = o_t
        
        return torch.cat((mem_f, mem_b), dim=1)
