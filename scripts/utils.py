import torch
import sklearn
import dgl
import errno
import json
import os, glob
import numpy as np
import pandas as pd
from functools import partial

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from dgl.data.utils import Subset
from dgllife.utils import WeaveAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph, EarlyStopping

from models import LocalRetro_model
from dataset import USPTODataset, USPTOTestDataset

def init_featurizer(args):
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    args['node_featurizer'] = WeaveAtomFeaturizer(atom_types = atom_types)
    args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
    return args

def get_configure(args):
    with open(args['config_path'], 'r') as f:
        config = json.load(f)
    global AtomTemplate_n, BondTemplate_n
    AtomTemplate_n = len(pd.read_csv('%s/atom_templates.csv' % args['data_dir']))
    BondTemplate_n = len(pd.read_csv('%s/bond_templates.csv' % args['data_dir']))
    args['AtomTemplate_n'], config['AtomTemplate_n'] = AtomTemplate_n, AtomTemplate_n
    args['BondTemplate_n'], config['BondTemplate_n'] = BondTemplate_n, BondTemplate_n
    config['in_node_feats'] = args['node_featurizer'].feat_size()
    config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    return config

def get_site_templates(args):
    with open('%s/site_template.json' % args['data_dir'], 'r') as f:
        args['site_templates'] = json.load(f)
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

def average_models(args):
    avg_model = None
    model_paths = [path for path in glob.glob('%s*.pth' % args['model_dir'])]
    print ('Averaging models...')
    model_paths = sorted(model_paths, key= lambda x: int(x.split('.pth')[0].split('cycle')[1]))
    for i, path in enumerate(model_paths):
        model = torch.load(path)
        model_weights = model['model_state_dict']
        if i == 0:
            avg_model = model_weights
        else:
            for (k, v) in avg_model.items():
                avg_model[k].mul_(i).add_(model_weights[k]).div_(i + 1)
    torch.save({'model_state_dict': avg_model}, '../models/LocalRetro_%s.pth' % args['dataset'])
    return 

def load_dataloader(args):
    if args['mode'] == 'train':
        dataset = USPTODataset(args, 
                            smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])

        train_set, val_set, test_set = Subset(dataset, dataset.train_ids), Subset(dataset, dataset.val_ids), Subset(dataset, dataset.test_ids)

        train_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], shuffle=True,
                                  collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        val_loader = DataLoader(dataset=val_set, batch_size=args['batch_size'],
                                collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs, num_workers=args['num_workers'])
        return train_loader, val_loader, test_loader
    else:
        test_set = USPTOTestDataset(args, 
                            smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                            node_featurizer=args['node_featurizer'],
                            edge_featurizer=args['edge_featurizer'])
        test_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'],
                                 collate_fn=collate_molgraphs_test, num_workers=args['num_workers'])
    return test_loader

def load_model(args):
    exp_config = get_configure(args)
    model = LocalRetro_model(
        node_in_feats=exp_config['in_node_feats'],
        edge_in_feats=exp_config['in_edge_feats'],
        node_out_feats=exp_config['node_out_feats'],
        edge_hidden_feats=exp_config['edge_hidden_feats'],
        num_step_message_passing=exp_config['num_step_message_passing'],
        attention_heads = exp_config['attention_heads'],
        attention_layers = exp_config['attention_layers'],
        AtomTemplate_n = exp_config['AtomTemplate_n'],
        BondTemplate_n = exp_config['BondTemplate_n'],
        activation = exp_config.get('activation', 'relu')
    )
    model = model.to(args['device'])
    print ('Parameters of loaded LocalRetro:')
    print (exp_config)

    if args['mode'] == 'train':
        loss_criterion = nn.CrossEntropyLoss(reduction = 'none')
        optimizer = Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=args['schedule_step'])
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args['schedule_step'], verbose=True)
        if os.path.exists(args['model_path']):
            user_answer = input('%s exists, want to (a) overlap (b) continue from checkpoint (c) make a new model?' % args['model_path'])
            if user_answer == 'a':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Overlap exsited model and training a new model...')
            elif user_answer == 'b':
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                stopper.load_checkpoint(model)
                print ('Train from exsited model checkpoint...')
            elif user_answer == 'c':
                model_name = input('Enter new model name: ')
                args['model_path'] = args['model_path'].replace('%s.pth' % args['dataset'], '%s.pth' % model_name)
                stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
                print ('Training a new model %s.pth' % model_name)
        else:
            stopper = EarlyStopping(mode = 'lower', patience=args['patience'], filename=args['model_path'])
        return model, loss_criterion, optimizer, scheduler, stopper
    
    else:
        model.load_state_dict(torch.load(args['model_path'], map_location=args['device'])['model_state_dict'])
        return model

def make_labels(graphs, labels, masks):
    atom_labels = []
    bond_labels = []
    for g, label, m in zip(graphs, labels, masks):
        n_atoms = g.number_of_nodes()
        n_bonds = g.number_of_edges()
        atom_label, bond_label = [0]*(n_atoms), [0]*(n_bonds-n_atoms)
        if m == 1:
            for l in label:
                label_type = l[0]
                label_idx = l[1]
                label_template = l[2]
                if label_type == 'a':
                    atom_label[label_idx] = label_template
                else:
                    bond_label[label_idx] = label_template
        atom_labels += atom_label
        bond_labels += bond_label   
    return torch.LongTensor(atom_labels), torch.LongTensor(bond_labels)

def flatten_list(t):
    return torch.LongTensor([item for sublist in t for item in sublist])

def collate_molgraphs(data):
    smiles, graphs, labels, masks = map(list, zip(*data))
    atom_labels, bond_labels = make_labels(graphs, labels, masks)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles, bg, atom_labels, bond_labels

def collate_molgraphs_test(data):
    smiles, graphs, rxns = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return smiles, bg, rxns

def predict(args, model, bg):
    bg = bg.to(args['device'])
    node_feats = bg.ndata.pop('h').to(args['device'])
    edge_feats = bg.edata.pop('e').to(args['device'])
    return model(bg, node_feats, edge_feats)
