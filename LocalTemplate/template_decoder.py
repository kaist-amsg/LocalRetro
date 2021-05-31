import os, re
import pandas as pd
from collections import defaultdict

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

from .template_includes_ring import include_ring_info

RDLogger.DisableLog('rdApp.*')

def match_edit_idx(prod_idx, temp_idx):
    prod2temp = {}
    if type(prod_idx) == type(0):
        prod2temp[prod_idx] = temp_idx
    else:
        for i, edit in enumerate(prod_idx):
            if type(edit) == type(0):
                prod2temp[edit] = temp_idx[i]
            else:
                for j, e in enumerate(edit):
                    prod2temp[e] = temp_idx[i][j]
    return prod2temp

def check_idx_match(mols):
    matched_idx = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                matched_idx[int(atom.GetProp('react_atom_idx'))] =  int(atom.GetProp('old_mapno'))
    return matched_idx

def match_subkeys(dict1, dict2):
    remained_check = len(dict1)
    for k in dict1.keys():
        if k not in dict2:
            return False
        if dict1[k] == dict2[k]:
            remained_check -= 1
    return remained_check == 0

def get_H(mol, matched_idx):
    atom_H = {}
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx in matched_idx.keys():
            atom_H[matched_idx[atom_idx]] = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
    return atom_H

def get_H_map(mol, temp, prod_idx, temp_idxs, H_change):
    
    matched_idx_list = []
    for temp_idx in temp_idxs:
        prod2temp = match_edit_idx(prod_idx, temp_idx)
        reaction = rdChemReactions.ReactionFromSmarts(temp)
        reactants = reaction.RunReactants([mol])
        for i, reactant in enumerate(reactants):
            matched_idx = check_idx_match(reactant)
            if match_subkeys(prod2temp, matched_idx):
                matched_idx_list.append(matched_idx)
            
    return matched_idx_list

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
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
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
    
def apply_template(products, template, edit_idx, temp_idx, H_change):
    mol = Chem.MolFromSmiles(products)
    matched_idx_list = get_H_map(mol, template, edit_idx, temp_idx, H_change)
    fit_templates = []
    right_reactants = []
    right_matched_idx = []
    for matched_idx in matched_idx_list:
        H_before = get_H(mol, matched_idx)
        H_after = {k: H_before[k] + change for k, change in H_change.items()}
        if not H_before:
            continue
        try:
            template = include_ring_info(products, template, edit_idx, temp_idx, matched_idx)
        except Exception as e:
            continue
            
        fit_template = fix_temp(template, H_before, H_after)
        fit_templates.append(fit_template)
        right_matched_idx.append(matched_idx)
        reaction = rdChemReactions.ReactionFromSmarts(fit_template)
        reactants = reaction.RunReactants([mol])
        right_reactant = select_right_reactant(matched_idx, reactants)
        if right_reactant:
            right_reactants += right_reactant
            
    right_reactants = list(set(right_reactants))
    fit_templates = list(set(fit_templates))
    return right_reactants, fit_templates, right_matched_idx
        
        
        