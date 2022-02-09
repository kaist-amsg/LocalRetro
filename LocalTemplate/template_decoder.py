import os, re, copy
import pandas as pd
from collections import defaultdict

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

RDLogger.DisableLog('rdApp.*')

chiral_type_map = {ChiralType.CHI_UNSPECIFIED : -1, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2}
chiral_type_map_inv = {v:k for k, v in chiral_type_map.items()}

a, b = 'a', 'b'

def get_isomers(smi):
    mol = Chem.MolFromSmiles(smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    isomers_smi = [Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers]
    return isomers_smi
    
def get_MaxFrag(smiles):
    return max(smiles.split('.'), key=len)
            
def isomer_match(preds, reac):
    reac_isomers = get_isomers(reac)
    for k, pred in enumerate(preds):
        try:
            pred_isomers = get_isomers(pred)
            if set(pred_isomers).issubset(set(reac_isomers)) or set(reac_isomers).issubset(set(pred_isomers)):
                return k+1
        except Exception as e:
            pass
    return -1
    
def get_edit_site(mol):
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

def get_idx_map(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    smiles = Chem.MolToSmiles(mol)
    num_map = {}
    for i, s in enumerate(smiles.split('.')):
        m = Chem.MolFromSmiles(s)
        for atom in m.GetAtoms():
            num_map[atom.GetAtomMapNum()] = atom.GetIdx()
    return num_map

def get_possible_map(pred_site, change_info):
    possible_maps = []
    if type(pred_site) == type(0):
        for edit_type, edits in change_info['edit_site'].items():
            if edit_type not in ['A', 'R']:
                continue
            for edit in edits:
                possible_maps.append({edit: pred_site})
    else:
        for edit_type, edits in change_info['edit_site'].items():
            if edit_type not in ['B', 'C']:
                continue
            for edit in edits:
                possible_maps.append({e:p for e, p in zip(edit, pred_site)})
    return possible_maps

def check_idx_match(mols, possible_maps):
    matched_maps = []
    found_map = {}
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno') and atom.HasProp('react_atom_idx'):
                found_map[int(atom.GetProp('old_mapno'))] = int(atom.GetProp('react_atom_idx'))
    for possible_map in possible_maps:
        if possible_map.items() <= found_map.items():
            matched_maps.append(found_map)
    return matched_maps

def fix_aromatic(mol):
    for atom in mol.GetAtoms():
        if not atom.IsInRing() and atom.GetIsAromatic():
            atom.SetIsAromatic(False)
        
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            bond.SetIsAromatic(False) 
            if str(bond.GetBondType()) == 'AROMATIC':
                bond.SetBondType(Chem.rdchem.BondType.SINGLE)
                
def validate_mols(mols):
    for mol in mols:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) == None:
            return False
    return True

def fix_reactant_atoms(product, reactants, matched_map, change_info):
    H_change, C_change, S_change = change_info['change_H'], change_info['change_C'], change_info['change_S']
    fixed_mols = []
    for mol in reactants:
        for atom in mol.GetAtoms():
            if atom.HasProp('old_mapno'):
                mapno = int(atom.GetProp('old_mapno'))
                if mapno not in matched_map:
                    return None
                product_atom = product.GetAtomWithIdx(matched_map[mapno])
                H_before = product_atom.GetNumExplicitHs() + product_atom.GetNumImplicitHs()
                C_before = product_atom.GetFormalCharge()
                S_before = chiral_type_map[product_atom.GetChiralTag()]
                H_after = H_before + H_change[mapno]
                C_after = C_before + C_change[mapno]
                S_after = S_change[mapno]
                if H_after < 0:
                    return None
                atom.SetNumExplicitHs(H_after)
                atom.SetFormalCharge(C_after)
                if S_after != 0:
                    atom.SetChiralTag(chiral_type_map_inv[S_after])
        fix_aromatic(mol)
        fixed_mols.append(mol)
    if validate_mols(fixed_mols):
        return tuple(fixed_mols)
    else:
        return None

def demap(mols, stereo = True):
    if type(mols) == type((0, 0)):
        ss = []
        for mol in mols:
            [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, stereo))
            if mol == None:
                return None
            ss.append(Chem.MolToSmiles(mol))
        return '.'.join(sorted(ss))
    else:
        [atom.SetAtomMapNum(0) for atom in mols.GetAtoms()]
        return '.'.join(sorted(Chem.MolToSmiles(mols, stereo).split('.')))
    
def read_prediction(smiles, prediction, atom_templates, bond_templates, template_infos, raw = False):
    mol = Chem.MolFromSmiles(smiles)
    if len(prediction) == 1:
        return mol, None, None, None, 0
    elif raw:
        edit_type, pred_site, pred_template_class, prediction_score = prediction # (edit_type, pred_site, pred_template_class)
    else:
        edit_type, pred_site, pred_template_class, prediction_score = eval(prediction) # (edit_type, pred_site, pred_template_class)
    atoms, bonds = get_edit_site(mol)
    idx_map = get_idx_map(mol)
    if edit_type == 'a':
        pred_site, template = atoms[pred_site], atom_templates[pred_template_class]
        if len(template.split('>>')[0].split('.')) > 1: pred_site = idx_map[pred_site] 
    else:
        pred_site, template = bonds[pred_site], bond_templates[pred_template_class]
        if len(template.split('>>')[0].split('.')) > 1: pred_site= (idx_map[pred_site[0]], idx_map[pred_site[1]])
    [atom.SetAtomMapNum(atom.GetIdx()) for atom in mol.GetAtoms()]
    if pred_site == None:
        return mol, pred_site, short_template, {}, 0
    return mol, pred_site, template, template_infos[template], prediction_score

def decode_localtemplate(product, pred_site, template, template_info):
    if pred_site == None:
        return None
    possible_maps = get_possible_map(pred_site, template_info)
    reaction = rdChemReactions.ReactionFromSmarts(template)
    reactants = reaction.RunReactants([product])
    decodes = []
    for output in reactants:
        if output == None:
            continue
        matched_maps = check_idx_match(output, possible_maps)
        for matched_map in matched_maps:
            decoded = fix_reactant_atoms(product, output, matched_map, template_info)
            if decoded == None:
                continue
            else:
                return demap(decoded)
    return None

