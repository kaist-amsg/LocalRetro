# multi-head attention code modified from https://github.com/SamLynnEvans/Transformer
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
    atom_pair_list = torch.transpose(sg.adjacency_matrix().coalesce().indices(), 0, 1)
    atom_pair_idx1 = atom_pair_list[:,0]
    atom_pair_idx2 = atom_pair_list[:,1]
    atom_pair_feats = torch.cat((node_feats[atom_pair_idx1], node_feats[atom_pair_idx2]), dim = 1)
    return atom_pair_feats

def unbatch_mask(bg, atom_feats, bond_feats):
    edit_feats = []
    masks = []
    feat_dim = atom_feats.size(-1)
    sg = bg.remove_self_loop()
    sg.ndata['h'] = atom_feats
    sg.edata['e'] = bond_feats
    gs = dgl.unbatch(sg, sg.batch_num_nodes(), (sg.batch_num_edges() - sg.batch_num_nodes()))
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
    gs = dgl.unbatch(sg, sg.batch_num_nodes(), (sg.batch_num_edges() - sg.batch_num_nodes()))
    atom_feats = []
    bond_feats = []
    for i, g in enumerate(gs):
        atom_feats.append(edit_feats[i][:g.num_nodes()])
        bond_feats.append(edit_feats[i][g.num_nodes():g.num_nodes()+g.num_edges()])
    return torch.cat(atom_feats, dim = 0), torch.cat(bond_feats, dim = 0)

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1).repeat(1,mask.size(-1),1)
        mask = mask.unsqueeze(1).repeat(1,scores.size(1),1,1)
        scores[~mask] = float(-9e15)
    scores = torch.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores) 
    output = torch.matmul(scores, v)
    return scores, output

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
        
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, mask=None):
        bs = x.size(0)
        k = self.k_linear(x).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(x).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(x).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores, output = attention(q, k, v, self.d_k, mask, self.dropout)
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = output + x
        output = self.layer_norm(output)
        return scores, output.squeeze(-1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_in*2) # position-wise
        self.w_2 = nn.Linear(d_in*2, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = output + x
        output = self.layer_norm(output)
        return output 

class MSA(nn.Module):
    def __init__(self, d_model, heads, n_layers = 1, dropout = 0.1):
        super(MSA, self).__init__()
        self.n_layers = n_layers
        self.att1 = MultiHeadAttention(heads, d_model, dropout)
        if n_layers > 1:
            att_stack = []
            pff_stack = []
            for _ in range(n_layers-1):
                att_stack.append(MultiHeadAttention(heads, d_model, dropout))
                pff_stack.append(PositionwiseFeedForward(d_model, dropout=dropout))
            self.att_stack = nn.ModuleList(att_stack)
            self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, mask):
        if self.n_layers > 1:
            for n in range(self.n_layers-1):
                _, x = self.att_stack[n](x, mask)
                x = self.pff_stack[n](x)
        score, output = self.att1(x, mask)
        
        return score, output