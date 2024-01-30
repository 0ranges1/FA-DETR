import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        logits = torch.bmm(q, k.transpose(1, 2))
        logits_scaled = logits / self.temperature # logits = qk/temperature, logits = (batchsize, len_q, len_k)
        attn = self.softmax(logits_scaled)
        
        prob_fg = logits_scaled.sigmoid()
        prob_fg = prob_fg.max(dim=-1, keepdim=True).values

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn, prob_fg


class SingleHeadSiameseAttention(nn.Module):
    """ Single-Head Attention Module. Weights for Q and K are shared in a Siamese manner. No proj weights for V."""
    def __init__(self, d_model):
        super().__init__()
        self.n_head = 1

        self.d_model = d_model
        self.d_head = d_model // self.n_head
        assert self.n_head * self.d_head == self.d_model

        self.w_qk = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        nn.init.xavier_uniform_(self.w_qk.weight)


        self.attention = ScaledDotProductAttention(temperature=np.power(self.d_head, 0.5)) # Q·K/ √d
        self.linear1 = nn.Linear(self.d_model, self.d_model)


    def forward(self, q, k, tsp): # query, class_prototype, window_encodings
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_tsp, _ = tsp.size()

        assert len_k == len_tsp
        residual = q # b x len_q x d_model

        prototype = k
        prototype = prototype.view(sz_b, len_k, self.n_head, self.d_head)
        prototype = prototype.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_head)  # (n_head * b) x len_k x d_head
        

        q = self.w_qk(q).view(sz_b, len_q, self.n_head, self.d_head)
        k = self.w_qk(k).view(sz_b, len_k, self.n_head, self.d_head)
        tsp = tsp.view(sz_b, len_tsp, self.n_head, self.d_head)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_head)  # (n_head * b) x len_q x d_head
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_head)  # (n_head * b) x len_k x d_head
        tsp = tsp.permute(2, 0, 1, 3).contiguous().view(-1, len_tsp, self.d_head)  # (n_head * b) x len_tsp x d_head
        

        # aggregate the window_encodings with the attention weights calculated by the query feature and class prototypes.
        tsp, attn, prob_fg = self.attention(q, k, tsp)
        tsp = tsp.view(self.n_head, sz_b, len_q, self.d_head)
        tsp = tsp.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x len_q x (n_head * d_head)

        
        prob_fg = prob_fg.view(self.n_head, sz_b, len_q, 1)
        prob_fg = prob_fg.mean(dim=0)
        output = (residual + tsp) * prob_fg
        
        output = self.linear1(output)

        return output #  b x len_q x d_model

        
