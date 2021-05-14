import numpy as np
import pandas as pd

import torch
import sklearn
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from functools import partial
from torch.utils.data import DataLoader

from utils import collate_molgraphs_test, load_LocalRetro, predict
from load_dataset import USPTO_test_Dataset
from get_edit import write_edits


def main(args, exp_config, test_set):
    exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs_test, num_workers=args['num_workers'])
    model = load_LocalRetro(exp_config)
    model = model.to(args['device'])
    stopper = EarlyStopping(filename = args['model_path']) 
    stopper.load_checkpoint(model)

    write_edits(args, model, test_loader, exp_config)
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('LocalRetro testing arguements')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-a', '--use-attention', default=True, help='Model use GRA or not')
    parser.add_argument('-k', '--top_num', default=100, help='Num. of predictions to write')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    args = init_featurizer(args)
    model_type = 'GRA' if args['use_attention'] else 'noGRA'
    args['result_path'] = '../results/%s_%s_outputs' % (args['dataset'], model_type)
    args['model_path'] = '../results/%s_%s_checkpoints/model.pth' % (args['dataset'], model_type)
    mkdir_p(args['result_path'])
    
    exp_config = get_configure()
    exp_config['ALRT_CLASS'] = len(pd.read_csv('../data/%s/atom_templates.csv' % args['dataset']))
    exp_config['BLRT_CLASS'] = len(pd.read_csv('../data/%s/bond_templates.csv' % args['dataset']))
    exp_config['attention_mode'] = model_type
    
    test_set = USPTO_test_Dataset(args,
                        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])

    main(args, exp_config, test_set)