from collections import defaultdict
import pandas as pd
import sys, os, re

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog('rdApp.*')

from local_template_extractor import extract_from_reaction

def remove_reagents(rxn):
    reactant = rxn.split('>>')[0]
    product = rxn.split('>>')[1]
    rs = reactant.split('.')
    ps = product.split('.')
    remove_frags = []
    for frag in ps:
        if frag in rs:
            remove_frags.append(frag)
    for frag in remove_frags:
        ps.remove(frag)
        rs.remove(frag)   
    return  '.'.join(sorted(rs)) + '>>' + '.'.join(sorted(ps))

def get_reaction_template(rxn, _id = 0):
    rxn = remove_reagents(rxn)
    rxn = {'reactants': rxn.split('>>')[0], 'products': rxn.split('>>')[1], '_id': _id}
    result = extract_from_reaction(rxn)
    return rxn, result

def dearomatic(template):
    for s in ['[c;', '[o;', '[n;', '[s;', '[c@']:
        template = template.replace(s, s.upper())
    return template

def fix_arom(mol):
    for atom in mol.GetAtoms():
        if not (atom.IsInRingSize(5) or atom.IsInRingSize(6)):
            atom.SetIsAromatic(False)
    return mol

def destereo(template):
    return template.replace('@', '')
        
def N_MolToSmiles(mol): # '[n:1]1[n:2][n:3][n:4][nH:5]1'
    smi = Chem.MolToSmiles(mol)
    if Chem.MolFromSmiles(smi) != None:
        pass
    else:
        prob_ns = re.findall('\[n:\d\]\d', smi)
        for n in prob_ns:
            nH = n.replace('[n', '[nH')
            smi = smi.replace(n, nH)
            if Chem.MolFromSmiles(smi) != None:
                break
    return smi

def clean_smarts(sma):
    mol = fix_arom(Chem.MolFromSmarts(sma.replace(';', '')))
    smi = Chem.MolToSmiles(mol)
    try:
        smi = demap(smi)
        return smi
    except:
        return sma
    

def demap(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

def reduce_template(temp):
    before_transform_smi = destereo(clean_smarts(temp.split('>>')[0]))
    before_transform_sma = destereo(temp.split('>>')[0])
    after_transform = temp.split('>>')[1]
    smi_template = before_transform_smi + '>>' + after_transform
    sma_template = before_transform_sma + '>>' + after_transform
    return smi_template, sma_template


def extract_templates(args):
    rxns = pd.read_csv('../data/%s/raw_train.csv' % args['dataset'])['reactants>reagents>production']
    
    class_train = '../data/%s/class_train.csv' % args['dataset']
    if os.path.exists(class_train):
        RXNHASCLASS = True
        rxn_class = pd.read_csv(class_train)['class']
        template_rxnclass = {i+1:set() for i in range(10)}
    else:
        RXNHASCLASS = False

    smiles2smarts = {}
    smiles2edit = {}
    smiles2Hs = {}
    atom_templates = defaultdict(int)
    bond_templates = defaultdict(int)
    unique_templates = set()
    
    for i, rxn in enumerate(rxns):
        if RXNHASCLASS:
            template_class = rxn_class[i]
        try:
            rxn, result = get_reaction_template(rxn, i)
            if 'reaction_smarts' not in result.keys():
                continue
            local_template = result['reaction_smarts']
            smi_template, sma_template = reduce_template(local_template)
            if smi_template not in smiles2smarts.keys(): # first come first serve
                template = sma_template
                smiles2smarts[smi_template] = sma_template
                smiles2edit[smi_template] = result['edit_sites'][2] # keep the map of changing idx
                smiles2Hs[smi_template] = result['H_change']
            else:
                template = smiles2smarts[smi_template]
                
            edit_sites = result['edit_sites'][0]
            
            atom_edit = False
            bond_edit = False
            
            for e in edit_sites:
                if type(e) == type(1):
                    atom_edit = True
                else:
                    bond_edit = True
            
            if atom_edit:
                atom_templates[template] += 1
            if bond_edit:
                bond_templates[template] += 1
            
            unique_templates.add(template)
            if RXNHASCLASS:
                template_rxnclass[template_class].add(template)
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except ValueError as e:
            print (i, e)
        if i % 100 == 0:
            print ('\r i = %s, # of atom template: %s, # of bond template: %s' % (i, len(atom_templates), len(bond_templates)), end='', flush=True)
    print ('\n total # of template: %s' %  len(unique_templates))
    derived_templates = {'atom':atom_templates, 'bond': bond_templates}
    
    if RXNHASCLASS:
        pd.DataFrame.from_dict(template_rxnclass, orient = 'index').T.to_csv('../data/%s/template_rxnclass.csv' % args['dataset'], index = None)
    smiles2smarts = pd.DataFrame({'Smiles_template': k, 'Smarts_template': t, 'edit_site':smiles2edit[k], 'change_H': smiles2Hs[k]} for k, t in smiles2smarts.items())
    smiles2smarts.to_csv('../data/%s/smiles2smarts.csv' % args['dataset'])
    
    return derived_templates

def export_template(derived_templates, args):
    for k in derived_templates.keys():
        local_templates = derived_templates[k]
        templates = []
        template_class = []
        template_freq = []
        sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
        c = 1
        for t in sorted_tuples:
            templates.append(t[0])
            template_freq.append(t[1])
            template_class.append(c)
            c += 1
        template_dict = {templates[i]:i+1  for i in range(len(templates)) }
        template_df = pd.DataFrame({'Template' : templates, 'Frequency' : template_freq, 'Class': template_class})

        template_df.to_csv('../data/%s/%s_templates.csv' % (args['dataset'], k))
    return


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('SimpleChiral Extractor')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-t', '--threshold', default=1,  help='Template refinement threshold')
    args = parser.parse_args().__dict__

    derived_templates = extract_templates(args)
    export_template(derived_templates, args)

    