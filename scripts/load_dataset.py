import os
import pandas as pd

from rdkit import Chem

import torch
import sklearn
import dgl.backend as F
from dgl.data.utils import save_graphs, load_graphs

class USPTODataset(object):
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('../data/%s/labeled_data.csv' % args['dataset'])
        self.train_ids = df.index[df['Split'] == 'train'].values
        self.val_ids = df.index[df['Split'] == 'val'].values
        self.test_ids = df.index[df['Split'] == 'test'].values
        self.smiles = df['Product'].tolist()
        self.templates = [int(t) for t in df['Template']]
        self.edit_idxs = [eval(e) for e in df['Edit_idx']]
        self.cache_file_path = '../data/saved_graphs/%s_dglgraph.bin' % args['dataset']
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            print('Processing dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('\rProcessing molecule %d/%d' % (i+1, len(self.smiles)), end='', flush=True)
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
            print ()
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
        return self.smiles[item], self.graphs[item], self.templates[item], self.edit_idxs[item]

    def __len__(self):
            return len(self.smiles)
        
class USPTO_test_Dataset(object):
    def __init__(self, args, smiles_to_graph, node_featurizer, edge_featurizer, load=True, log_every=1000):
        df = pd.read_csv('../data/%s/raw_test.csv' % args['dataset'])
        self.rxns = df['reactants>reagents>production'].tolist()
        self.smiles = [rxn.split('>>')[-1] for rxn in self.rxns]
        self.cache_file_path = '../data/saved_graphs/%s_test_dglgraph.bin' % args['dataset']
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer, load, log_every)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer, load, log_every):
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved test dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
        else:
            print('Processing test dgl graphs from scratch...')
            self.graphs = []
            for i, s in enumerate(self.smiles):
                if (i + 1) % log_every == 0:
                    print('Processing molecule %d/%d' % (i+1, len(self.smiles)))
                self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                   edge_featurizer=edge_featurizer, canonical_atom_order=False))
                
            save_graphs(self.cache_file_path, self.graphs)

    def __getitem__(self, item):
            return self.smiles[item], self.graphs[item], self.rxns[item]

    def __len__(self):
            return len(self.smiles)