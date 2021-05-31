import torch
import sklearn
import dgl
import errno
import json
import os
import numpy as np
import torch.nn.functional as F

from dgl.data.utils import Subset
from models import LocalRetro

def init_featurizer(args):
    from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer
    args['node_featurizer'] = WeaveAtomFeaturizer()
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    return args

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise

def split_dataset(args, dataset):
    return Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

def get_configure():
    with open('../data/config.json', 'r') as f:
        config = json.load(f)
    return config

def collate_atom_labels(graphs, edit_atoms, templates):
    labels = []
    for i, g in enumerate(graphs):
        template = templates[i]
        edit_atom = edit_atoms[i]
        if len(edit_atom) == 1:
            edit_atom = edit_atom[0]
            if type(edit_atom) == type(1):
                for node in g.nodes():
                    if int(node) == edit_atom and template!= 0:
                        labels.append(template)
                    else:
                        labels.append(0)
            else:
                labels += [0]*g.num_nodes()
        else:
            for node in g.nodes():
                if int(node) in edit_atom and template!= 0:
                    labels.append(template)
                else:
                    labels.append(0)
    return torch.LongTensor(labels)

def collate_bond_labels(graphs, edit_atoms, templates):
    labels = []
    for i, g in enumerate(graphs):
        g = g.remove_self_loop()
        template = templates[i]
        atom_pairs = g.adjacency_matrix().coalesce().indices().numpy().T
        edit_atom = edit_atoms[i]
        if len(edit_atom) == 1: # single edit
            edit_atom = edit_atom[0]
            if type(edit_atom) == type(1): # atom edit
                labels += [0]*len(atom_pairs) 
            elif type(edit_atom) == type((0,1)): # one bond edit
                for ap in atom_pairs:
                    ap = tuple(ap)
                    if ap[0] == ap[1]:
                        continue
                    if ap == edit_atom:
                        labels.append(template)
                    else:
                        labels.append(0)                    
        else: # multi-bond edit
            for ap in atom_pairs:
                ap = tuple(ap)
                if ap[0] == ap[1]:
                    continue
                if ap in edit_atom:
                    labels.append(template)
                else:
                    labels.append(0)
    return torch.LongTensor(labels)

def collate_molgraphs(data):
    smiles, graphs, templates, edit_atoms = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    atom_labels = collate_atom_labels(graphs, edit_atoms, templates)
    bond_labels = collate_bond_labels(graphs, edit_atoms, templates)
    return smiles, bg, atom_labels, bond_labels


def collate_molgraphs_test(data):
    smiles, graphs, rxns = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles, bg, rxns

def load_LocalRetro(exp_configure):
    model = LocalRetro(
        node_in_feats=exp_configure['in_node_feats'],
        edge_in_feats=exp_configure['in_edge_feats'],
        node_out_feats=exp_configure['node_out_feats'],
        edge_hidden_feats=exp_configure['edge_hidden_feats'],
        num_step_message_passing=exp_configure['num_step_message_passing'],
        use_GRA = exp_configure['use_GRA'],
        attention_heads = exp_configure['attention_heads'],
        ALRT_CLASS = exp_configure['ALRT_CLASS'],
        BLRT_CLASS = exp_configure['BLRT_CLASS'])
    return model


def predict(args, model, bg):
    bg = bg.to(args['device'])
    node_feats = bg.ndata.pop('h').to(args['device'])
    edge_feats = bg.edata.pop('e').to(args['device'])
    return model(bg, node_feats, edge_feats)
