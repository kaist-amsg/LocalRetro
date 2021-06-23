import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN
from model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, MSA

class LocalRetro(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 AtomTemplate_n, 
                 BondTemplate_n,
                 GRA):
        super(LocalRetro, self).__init__()
        
        self.GRA = GRA
        
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.linearB = nn.Linear(node_out_feats*2, node_out_feats)

        self.att = MSA(node_out_feats, attention_heads, attention_layers)
        
        self.atom_editor =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, AtomTemplate_n+1))
        self.bond_editor =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, BondTemplate_n+1))

    def forward(self, g, node_feats, edge_feats):

        node_feats = self.mpnn(g, node_feats, edge_feats)
        atom_feats1 = node_feats
        bond_feats1 = self.linearB(pair_atom_feats(g, node_feats))
        
        if self.GRA:
            edit_feats, mask = unbatch_mask(g, atom_feats1, bond_feats1)
            attention_score, edit_feats = self.att(edit_feats, mask)
        else:
            edit_feats, mask = unbatch_mask(g, atom_feats1, bond_feats1)
            attention_score = None
            
        atom_feats2, bond_feats2 = unbatch_feats(g, edit_feats)
        atom_out = self.atom_editor(atom_feats2) 
        bond_out = self.bond_editor(bond_feats2) 

        return atom_out, bond_out, attention_score


