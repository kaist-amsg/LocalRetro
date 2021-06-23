import os, sys, re
import pandas as pd
from argparse import ArgumentParser

import rdkit 
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
RDLogger.DisableLog('rdApp.*')  
sys.path.append('../')

from LocalTemplate.template_extractor import extract_from_reaction
from Extract_from_train_data import reduce_template, get_reaction_template

def matchwithtemp(template1, template2, replacement_dict):
    template1 = template1.split('>>')[1]
    template2 = template2.split('>>')[1]
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

def match_num(edit_idx, replace_dict):
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

def get_idx_map(product_smiles, replace_dict):
    mol = Chem.MolFromSmiles(product_smiles)
    idx_map = {}
    for atom in mol.GetAtoms():
        idx = str(atom.GetIdx())
        atom_map = str(atom.GetAtomMapNum())
        idx_map[atom_map] = idx
    return {i:idx_map[k] for k, i in replace_dict.items()}
    
def get_edit_site(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        bonds = atom.GetBonds()
        for bond in bonds:
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    return A, B

def labeling_dataset(args, split, template_dicts, smiles2smarts, smiles2edit):
    
    if os.path.exists('../data/%s/preprocessed_%s.csv' % (args['dataset'], split)):
        print ('%s data already preprocessed...loaded data!' % split)
        return pd.read_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    
    atom_templates = template_dicts['atom']
    bond_templates = template_dicts['bond']
    rxns = pd.read_csv('../data/%s/raw_%s.csv' % (args['dataset'], split))['reactants>reagents>production']
    products = []
    atom_labels = []
    bond_labels = []
    success = 0
    for n, rxn in enumerate(rxns):
        product = rxn.split('>>')[1]
        try:
            rxn, result = get_reaction_template(rxn, n)
            local_template = result['reaction_smarts']
            smi_template, sma_template = reduce_template(local_template)
            if smi_template not in smiles2smarts.keys():
                products.append(product)
                atom_labels.append(0)
                bond_labels.append(0)
                continue
            else:
                replace_dict = matchwithtemp(sma_template, smiles2smarts[smi_template], result['replacement_dict']) 
                replace_dict = get_idx_map(product, replace_dict)
                edit_sites = eval(match_num(smiles2edit[smi_template], replace_dict))
                local_template = smiles2smarts[smi_template]
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            products.append(product)
            atom_labels.append(0)
            bond_labels.append(0)
            continue
            
        if len(edit_sites) <= args['max_edit_n']:
            atom_sites, bond_sites = get_edit_site(product)
            try:
                if local_template not in  atom_templates.keys() and local_template not in  bond_templates.keys():
                    products.append(product)
                    atom_labels.append(0)
                    bond_labels.append(0)
                else:
                    atom_label = [0] * len(atom_sites)
                    bond_label = [0] * len(bond_sites)
                    for edit_site in edit_sites:
                        if type(edit_site) == type(1):
                            atom_label[atom_sites.index(edit_site)] = atom_templates[local_template]
                        else:
                            bond_label[bond_sites.index(edit_site)] = bond_templates[local_template]
                    products.append(product)
                    atom_labels.append(atom_label)
                    bond_labels.append(bond_label)
                    success += 1
            except Exception as e:
                products.append(product)
                atom_labels.append(0)
                bond_labels.append(0)
                continue
                
            if n % 100 == 0:
                print ('\r Processing %s %s data..., success %s data (%s/%s)' % (args['dataset'], split, success, n, len(rxns)), end='', flush=True)
        else:
            print ('\nReaction # %s has too many (%s) edits... may be wrong mapping!' % (n, len(edit_sites)))
            products.append(product)
            atom_labels.append(0)
            bond_labels.append(0)
            
    print ('\nDerived tempaltes cover %.3f of %s data reactions' % ((success/len(rxns)), split))
    df = pd.DataFrame({'Products': products, 'Atom_label': atom_labels, 'Bond_label': bond_labels, 'Reaction': rxns})
    df.to_csv('../data/%s/preprocessed_%s.csv' % (args['dataset'], split))
    return df

def combine_preprocessed_data(train_pre, val_pre, test_pre, args):
    train_valid = train_pre[train_pre['Atom_label'] != 0].reset_index()
    val_valid = val_pre[val_pre['Atom_label'] != 0].reset_index()
    test_valid = test_pre[test_pre['Atom_label'] != 0].reset_index()
    
    train_valid['Split'] = ['train'] * len(train_valid)
    val_valid['Split'] = ['val'] * len(val_valid)
    test_valid['Split'] = ['test'] * len(test_valid)

    all_valid = train_valid.append(val_valid, ignore_index=True)
    all_valid = all_valid.append(test_valid, ignore_index=True)
    print ('Valid data size: %s' % len(all_valid))
    all_valid.to_csv('../data/%s/labeled_data.csv' % args['dataset'], index = None)
    return

def load_template_dict(args):
    template_dicts = {}
    for site in ['atom', 'bond']:
        template_df = pd.read_csv('../data/%s/%s_templates.csv' % (args['dataset'], site))
        template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index  if template_df['Frequency'][i] >= args['threshold']}
        print ('loaded %s %s templates' % (len(template_dict), site))
        template_dicts[site] = template_dict
    return template_dicts

def load_smi2sma_dict(args):
    template_df = pd.read_csv('../data/%s/smiles2smarts.csv' % args['dataset'])
    smiles2smarts = {template_df['Smiles_template'][i]:template_df['Smarts_template'][i] for i in template_df.index}
    smiles2edit = {template_df['Smiles_template'][i]:template_df['edit_site'][i] for i in template_df.index}
    return smiles2smarts, smiles2edit

if __name__ == '__main__':
    parser = ArgumentParser('Local Template Preprocessing')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-t', '--threshold', default=1,  help='Template refinement threshold')
    parser.add_argument('-m', '--max-edit-n', default=8,  help='Maximum number of edit number')
    args = parser.parse_args().__dict__

    template_dicts = load_template_dict(args)
    smiles2smarts, smiles2edit = load_smi2sma_dict(args)

    train_pre = labeling_dataset(args, 'train', template_dicts, smiles2smarts, smiles2edit)
    val_pre = labeling_dataset(args, 'val', template_dicts, smiles2smarts, smiles2edit)
    test_pre = labeling_dataset(args, 'test', template_dicts, smiles2smarts, smiles2edit)
    
    combine_preprocessed_data(train_pre, val_pre, test_pre, args)
    
        
        
        