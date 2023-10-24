import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import math
import sklearn
import dgl
import dgllife

def pair_atom_feats(g, node_feats):
    sg = g.remove_self_loop() # in case g includes self-loop
    atom_idx1, atom_idx2 = sg.edges()
    atom_pair_feats = torch.cat((node_feats[atom_idx1.long()], node_feats[atom_idx2.long()]), dim = 1)
    return atom_pair_feats

def unbatch_mask(bg, atom_feats, bond_feats):
    edit_feats = []
    masks = []
    feat_dim = atom_feats.size(-1)
    sg = bg.remove_self_loop()
    sg.ndata['h'] = atom_feats
    sg.edata['e'] = bond_feats
    gs = dgl.unbatch(sg)
    for g in gs:
        e_feats = torch.cat((g.ndata['h'], g.edata['e']), dim = 0)
        mask = torch.ones(e_feats.size()[0], dtype=torch.uint8)
        edit_feats.append(e_feats)
        masks.append(mask)
    
    edit_feats = pad_sequence(edit_feats, batch_first=True, padding_value= 0)
    masks = pad_sequence(masks, batch_first=True, padding_value= 0)
    
    return edit_feats, masks 
    
def unbatch_feats(bg, edit_feats):
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg)
    atom_feats = []
    bond_feats = []
    for i, g in enumerate(gs):
        atom_feats.append(edit_feats[i][:g.num_nodes()])
        bond_feats.append(edit_feats[i][g.num_nodes():g.num_nodes()+g.num_edges()])
    return torch.cat(atom_feats, dim = 0), torch.cat(bond_feats, dim = 0)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.reset_parameters()
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1,mask.size(-1),1)
            mask = mask.unsqueeze(1).repeat(1,scores.size(1),1,1)
            scores[~mask.bool()] = float(-9e15)
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores) 
        output = torch.matmul(scores, v)
        return scores, output

    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores, output = self.attention(q, k, v, mask)
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)
    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, activation=GELU(), dropout = 0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            activation,
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, x):
        output = self.net(x)
        return self.layer_norm(x + output)

class Global_Reactivity_Attention(nn.Module):
    def __init__(self, d_model, heads, n_layers = 1, dropout = 0.1, activation=GELU()):
        super(Global_Reactivity_Attention, self).__init__()
        self.n_layers = n_layers
        att_stack = []
        pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, activation, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, mask):
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x