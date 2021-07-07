import os, re
import pandas as pd
from collections import defaultdict

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

from .template_includes_ring import include_ring_info

RDLogger.DisableLog('rdApp.*')

def match_edit_idx(pred_idx, temp_idx):
    temp2pred = {}
    if type(pred_idx) == type(0):
        temp2pred[temp_idx] = pred_idx
    else:
        for i, edit in enumerate(pred_idx):
            if type(edit) == type(0):
                temp2pred[temp_idx[i]] = edit
            else:
                for j, e in enumerate(edit):
                    temp2pred[temp_idx[i][j]] = e
    return temp2pred

def check_idx_match(mols):
    matched_idx = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                matched_idx[int(atom.GetProp('old_mapno'))] =  int(atom.GetProp('react_atom_idx'))
    return matched_idx

def match_subkeys(dict1, dict2):
    unchecked = len(dict1)
    for k in dict1.keys():
        if k not in dict2:
            return False
        if dict1[k] == dict2[k]:
            unchecked -= 1
    return unchecked == 0

def get_H1(mol, matched_idx, H_change):
    H_before = {}
    H_after = {}
    matched_idx_reverse = {v:k for k,v in matched_idx.items()}
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx in matched_idx.values():
            mapno = matched_idx_reverse[atom_idx]
            atom_H = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            H_before[mapno] = atom_H
            H_after[mapno] = atom_H + H_change[mapno]
    return H_before, H_after, matched_idx_reverse

## If there are more than two products in the reaction
def get_H2(mols, H_change):
    H_before = {}
    H_after = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                mapno = int(atom.GetProp('old_mapno'))
                try:
                    atom_H = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
                    H_before[mapno] = atom_H
                    H_after[mapno] = atom_H + H_change[mapno]
                except:
                    atom.UpdatePropertyCache(strict = False)
                    atom_H = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
                    H_before[mapno] = atom_H - H_change[mapno]
                    H_after[mapno] = atom_H
                
    return H_before, H_after

def match_pred_idx(products, temp, prod_idx, temp_idxs):
    mol = Chem.MolFromSmiles(products)
    matched_idx_list = []
    matched_reactants_list = []
    for temp_idx in temp_idxs:
        try:
            temp2pred = match_edit_idx(prod_idx, temp_idx)
            reaction = rdChemReactions.ReactionFromSmarts(temp)
            ms = [Chem.MolFromSmiles(p) for p in products.split('.')]
            reactants = reaction.RunReactants(ms)
            if len(reactants) == 0:
                reactants = reaction.RunReactants(ms[::-1])
            for i, reactant in enumerate(reactants):
                matched_idx = check_idx_match(reactant)
                if match_subkeys(temp2pred, matched_idx):
                    matched_idx_list.append(matched_idx)
                    matched_reactants_list.append(reactant)
        except Exception as e:
#             print (e)
            pass
    return matched_idx_list, matched_reactants_list

def fix_smart(smarts, H_map):
    temp_atoms = re.findall(r"\[.*?]", smarts)
    for a in temp_atoms:
        if ':' not in a:
            continue
        num = int(a.split(':')[1].split(']')[0])
        if num not in H_map:
            pass
        else:
            b = a.replace(';', ';H%s;' % H_map[num])
            smarts = smarts.replace(a, b)
    return smarts

def fix_temp(temp, H_before, H_after):
    sma1, sma2 = temp.split('>>')[0], temp.split('>>')[1]
    return fix_smart(sma1, H_before) + '>>' + fix_smart(sma2,  H_after)

    
def fix_aromatic(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
        
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False) 
            if str(bond.GetBondType()) == 'AROMATIC':
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)

def deradical(mol):
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + atom.GetNumRadicalElectrons())
            atom.SetNumRadicalElectrons(0)
    return mol

def get_stereo(mol):
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    Chem.DetectBondStereoChemistry(mol, conf)
    Chem.SetBondStereoFromDirections(mol)
    return 

def demap(mol):
    fix_aromatic(mol)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    smi = Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(deradical(Chem.MolFromSmiles(smi)))

def match_subdict(parent, child):
    match = False
    for k,v in child.items():
        if k not in parent.keys():
            return match
        elif parent[k] != v:
            return match
    return True
    
def select_right_reactant(matched_idx, reactants):
    right_reactants = []
    all_reactants = []
    for reactant in reactants:
        try:
            reactants_smi = '.'.join(sorted([demap(r) for r in reactant]))
            all_reactants.append(reactants_smi)

            if match_subdict(check_idx_match(reactant), matched_idx):
                right_reactants.append(reactants_smi)
        except Exception as e:
#             print (e)
            pass
    
    right_reactants = list(set(right_reactants))
    return right_reactants
    
def get_idx_map(products):
    mol = Chem.MolFromSmiles(products)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    num_map = {}
    for i, s in enumerate(smiles.split('.')):
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            num_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return num_map

def apply_template(products, template, edit_idx, temp_idx, H_change):
    mol = Chem.MolFromSmiles(products)
    matched_idx_list, matched_reactants_list = match_pred_idx(products, template, edit_idx, temp_idx)
    fit_templates = []
    right_reactants = []
    right_matched_idx = []
    for matched_idx, matched_reactants in zip(matched_idx_list, matched_reactants_list):
        if len(template.split('>>')[0].split('.')) == 1:
            H_before, H_after, matched_idx_reverse = get_H1(mol, matched_idx, H_change)
            try:
                template = include_ring_info(products, template, edit_idx, temp_idx, matched_idx_reverse)
            except Exception as e:
#                 print (e)
                continue
        else:
            H_before, H_after = get_H2(matched_reactants, H_change)
            
        if not H_before:
            continue     
        fit_template = fix_temp(template, H_before, H_after)
        reaction = rdChemReactions.ReactionFromSmarts(fit_template)
        ms = [Chem.MolFromSmiles(p) for p in products.split('.')]
        reactants = reaction.RunReactants(ms)
        if len(reactants) == 0:
            reactants = reaction.RunReactants(ms[::-1])
        right_reactant = select_right_reactant(matched_idx, reactants)
        if right_reactant:
            fit_templates.append(fit_template)
            right_matched_idx.append(matched_idx)
            right_reactants += right_reactant
            
    right_reactants = list(set(right_reactants))
    fit_templates = list(set(fit_templates))
    return right_reactants, fit_templates, right_matched_idx
        
        
        