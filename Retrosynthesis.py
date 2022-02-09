import json
import pandas as pd

import torch
from torch import nn
import sklearn

from rdkit.Chem import PandasTools, AllChem

import dgl
from dgllife.utils import smiles_to_bigraph, WeaveAtomFeaturizer, CanonicalBondFeaturizer
from functools import partial

from scripts.utils import init_featurizer, load_model, collate_molgraphs_test
from scripts.get_edit import combined_edit
from LocalTemplate.template_decoder import *

def predict(model, graph, device):
    bg = dgl.batch([graph])
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    bg = bg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    edge_feats = bg.edata.pop('e').to(device)
    return model(bg, node_feats, edge_feats)

def load_templates(args):
    atom_templates = pd.read_csv('%s/atom_templates.csv' % args['data_dir'])
    bond_templates = pd.read_csv('%s/bond_templates.csv' % args['data_dir'])
    template_infos = pd.read_csv('%s/template_infos.csv' % args['data_dir'])

    atom_templates = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
    bond_templates = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
    template_infos = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i]), 'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}
    return atom_templates, bond_templates, template_infos

def init_LocalRetro(args):
    args['mode'] = 'test'
    args = init_featurizer(args)
    model = load_model(args)
    atom_templates, bond_templates, template_infos = load_templates(args)
    smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs']
    node_featurizer = WeaveAtomFeaturizer(atom_types = atom_types)
    edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
    graph_function = lambda s: smiles_to_graph(s, node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, canonical_atom_order = False)
    return model, graph_function, atom_templates, bond_templates, template_infos


def remap(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    
def retrosnythesis(smiles, model, graph_function, device, atom_templates, bond_templates, template_infos, top_k = 10, verbose = False):
    model.eval()
    graph = graph_function(smiles)
    with torch.no_grad():
        atom_logits, bond_logits, _ = predict(model, graph, device)
        atom_logits = nn.Softmax(dim=1)(atom_logits)
        bond_logits = nn.Softmax(dim=1)(bond_logits)
        graph = graph.remove_self_loop()
        pred_types, pred_sites, pred_scores = combined_edit(graph, atom_logits, bond_logits, top_k)

    predictions = [(pred_types[k], pred_sites[k][0], pred_sites[k][1], pred_scores[k]) for k in range(top_k)]
    predicted_reactants = [smiles]
    predicted_sites = [None]
    predicted_templates = [None]
    predicted_scores = [None]
    for k, prediction in enumerate(predictions):
        mol, pred_site, template, template_info, score = read_prediction(smiles, prediction, atom_templates, bond_templates, template_infos, True)
        local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
        decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
        if verbose:
            print ('top %s' % (k+1), pred_site, local_template, score)
        try:
            decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
            if decoded_smiles == None or decoded_smiles in predicted_reactants:
                continue
        except Exception as e:
            print (e)
            continue
        predicted_sites.append(pred_site)
        predicted_reactants.append(decoded_smiles)
        predicted_templates.append(local_template)
        predicted_scores.append(score)
                
    results_df = pd.DataFrame({'SMILES': predicted_reactants, 'Predicted site': predicted_sites, 'Local reaction template': predicted_templates, 'Score': predicted_scores})
    PandasTools.AddMoleculeColumnToFrame(results_df,'SMILES','Molecule')
    remap(results_df['Molecule'][0])
    return results_df
