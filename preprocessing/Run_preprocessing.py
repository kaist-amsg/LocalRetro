import os, sys, re
import pandas as pd

import rdkit 
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog('rdApp.*')  
sys.path.append('../')

from LocalTemplate.template_extractor import extract_from_reaction
from Extract_from_train_data import reduce_template, get_reaction_template

def matchwithtemp(template1, template2, replacement_dict):
    matched_idx = {}
    for char1, char2 in zip(template1, template2):
        if char1 == char2:
            pass
        else:
            matched_idx[char1] = char2
            
    new_replacement_dict = {}
    for k in replacement_dict.keys():
        if replacement_dict[k] not in matched_idx.keys():
            new_replacement_dict[k] = replacement_dict[k]
        else:
            new_replacement_dict[k] = matched_idx[replacement_dict[k]]
    return new_replacement_dict

def match_num_(edit_idx, replace_dict):
    new_edit_idx = ''
    idx = ''
    for s in str(edit_idx):
        if s.isdigit():
            idx += s
        else:
            if idx.isdigit():
                new_edit_idx += replace_dict[idx]
            new_edit_idx += s
            idx = ''
    return new_edit_idx

def match_num(a, b, edit_idx, replace_dict):
    n_dict = {}
    a = a.split('>>')[0]
    b = b.split('>>')[0]
    mola = Chem.MolFromSmarts(a)
    molb = Chem.MolFromSmarts(b)
    for atoma, atomb in zip(mola.GetAtoms(), molb.GetAtoms()):
        n_dict[str(atoma.GetAtomMapNum())] = replace_dict[str(atomb.GetAtomMapNum())]
    
    new_edit_idx = ''
    idx = ''
    for s in str(edit_idx):
        if s.isdigit():
            idx += s
        else:
            if idx.isdigit():
                new_edit_idx += n_dict[idx]
            new_edit_idx += s
            idx = ''
    return new_edit_idx

def get_idx_map(product_smiles, replace_dict):
    mol = Chem.MolFromSmiles(product_smiles)
    idx_map = {}
    for atom in mol.GetAtoms():
        idx = str(atom.GetIdx())
        atom_map = str(atom.GetAtomMapNum())
        idx_map[atom_map] = idx
    return {i:idx_map[k] for k, i in replace_dict.items()}
    
def labeling_dataset(args, split, atom_templates, bond_templates, smiles2smarts, smiles2edit):
    if os.path.exists('../data/%s/preprocessed_%s.csv' % (args['dataset'], split)):
        print ('%s data already preprocessed...loaded data!' % split)
        return pd.read_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    
    rxns = pd.read_csv('../data/%s/raw_%s.csv' % (args['dataset'], split))['reactants>reagents>production']
    products = []
    edit_idx = []
    template_classes = []
    success = 0
    for n, rxn in enumerate(rxns):
        product = rxn.split('>>')[1]
        try:
            rxn, result = get_reaction_template(rxn, n)
            local_template = result['reaction_smarts']
            smi_template, sma_template = reduce_template(local_template)
            if smi_template not in smiles2smarts.keys(): # first come first serve
                products.append(rxn['products'])
                edit_idx.append(-1)
                template_classes.append(0)
                continue
            else:
                replace_dict = matchwithtemp(sma_template, smiles2smarts[smi_template], result['replacement_dict']) 
                replace_dict = get_idx_map(rxn['products'], replace_dict)
                edit_sites = eval(match_num_(smiles2edit[smi_template], replace_dict))
                local_template = smiles2smarts[smi_template]
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print (i, e)
            products.append(product)
            edit_idx.append(-1)
            template_classes.append(0)
            continue
        if len(edit_sites) < 5:
            if local_template in atom_templates.keys():
                template_class = atom_templates[local_template]
                products.append(rxn['products'])
                edit_idx.append(edit_sites)
                template_classes.append(template_class)
                success += 1
            elif local_template in bond_templates.keys():
                template_class = bond_templates[local_template]
                products.append(product)
                edit_idx.append(edit_sites)
                template_classes.append(template_class)
                success += 1
            else:
                products.append(product)
                edit_idx.append(-1)
                template_classes.append(0)
            if n % 100 == 0:
                print ('\r Processing %s %s data..., success %s data (%s/%s)' % (args['dataset'], split, success, n, len(rxns)), end='', flush=True)
        else:
            print ('\nReaction # %s has too many edits...may be wrong mapping!' % n)
            products.append(rxn['products'])
            edit_idx.append(-1)
            template_classes.append(0)
            
    print ('\nDerived tempaltes cover %.3f of %s data reactions' % ((success/len(rxns)), split))
    df = pd.DataFrame({'Product': products, 'Edit_idx': edit_idx, 'Template': template_classes, 'Reaction': rxns})
    df.to_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    return df

def combine_preprocessed_data(train_pre, val_pre, test_pre, args):
    train_valid = train_pre[train_pre['Template'] != 0].reset_index()
    val_valid = val_pre[val_pre['Template'] != 0].reset_index()
    test_valid = test_pre[test_pre['Template'] != 0].reset_index()
    
    train_valid['Split'] = ['train'] * len(train_valid)
    val_valid['Split'] = ['val'] * len(val_valid)
    test_valid['Split'] = ['test'] * len(test_valid)

    all_valid = train_valid.append(val_valid, ignore_index=True)
    all_valid = all_valid.append(test_valid, ignore_index=True)
    print ('Valid data size: %s' % len(all_valid))
    all_valid.to_csv('../data/%s/labeled_data.csv' % args['dataset'], index = None)
    return

def load_template_dict(path):
    template_df = pd.read_csv(path)
    template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index}
    return template_dict

def load_smi2sma_dict(path):
    template_df = pd.read_csv(path)
    smiles2smarts = {template_df['Smiles_template'][i]:template_df['Smarts_template'][i] for i in template_df.index}
    smiles2edit = {template_df['Smiles_template'][i]:template_df['edit_site'][i] for i in template_df.index}
    return smiles2smarts, smiles2edit

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('SimpleChiral Extractor')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-t', '--threshold', default=1,  help='Template refinement threshold')
    args = parser.parse_args().__dict__

    atom_templates = load_template_dict('../data/%s/atom_templates.csv' % args['dataset'])
    bond_templates = load_template_dict('../data/%s/bond_templates.csv' % args['dataset'])
    smiles2smarts, smiles2edit = load_smi2sma_dict('../data/%s/smiles2smarts.csv' % args['dataset'])

    train_pre = labeling_dataset(args, 'train', atom_templates, bond_templates, smiles2smarts, smiles2edit)
    val_pre = labeling_dataset(args, 'val', atom_templates, bond_templates, smiles2smarts, smiles2edit)
    test_pre = labeling_dataset(args, 'test', atom_templates, bond_templates, smiles2smarts, smiles2edit)
    
    
    combine_preprocessed_data(train_pre, val_pre, test_pre, args)
    
        
        
        