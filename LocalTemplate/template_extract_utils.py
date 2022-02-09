import re
import copy
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import ChiralType

chiral_type_map = {ChiralType.CHI_UNSPECIFIED : 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2}
bond_type_map = {'SINGLE': '-', 'DOUBLE': '=', 'TRIPLE': '#', 'AROMATIC': '@'}

def get_template_bond(temp_order, bond_smarts):
    bond_match = {}
    for n, _ in enumerate(temp_order):
        bond_match[(temp_order[n], temp_order[n-1])] = bond_smarts[n-1]
        bond_match[(temp_order[n-1], temp_order[n])] = bond_smarts[n-1]
    return bond_match   

def bond_to_smiles(bond):
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])
    bond_smarts = bond_type_map[str(bond.GetBondType())]
    return '{}{}{}'.format(atoms[0], bond_smarts, atoms[1])

def check_bond_break(bond1, bond2):
    if bond1 == None and bond2 != None:
        return False
    elif bond1 != None and bond2 == None:
        return True
    else:
        return False

def check_bond_formed(bond1, bond2):
    if bond1 != None and bond2 == None:
        return False
    elif bond1 == None and bond2 != None:
        return True
    else:
        return False
    
def check_bond_change(pbond, rbond):
    if pbond == None or rbond == None:
        return False
    elif bond_to_smiles(pbond) != bond_to_smiles(rbond):
        return True
    else:
        return False
    
def atom_neighbors(atom):
    neighbor = []
    for n in atom.GetNeighbors():
        neighbor.append(n.GetAtomMapNum())
    return sorted(neighbor)

def extend_changed_atoms(changed_atom_tags, reactants, max_map):
    for reactant in reactants:
        extend_idx = []
        for atom in reactant.GetAtoms():
            if str(atom.GetAtomMapNum()) in changed_atom_tags:
                for n in atom.GetNeighbors():
                    if n.GetAtomMapNum() == 0:
                        extend_idx.append(n.GetIdx())
        for idx in extend_idx:
            reactant.GetAtomWithIdx(idx).SetAtomMapNum(max_map)

def check_atom_change(patom, ratom):
    return atom_neighbors(patom) != atom_neighbors(ratom)
    
def label_retro_edit_site(products, reactants, edit_num):
    edit_num = [int(num) for num in edit_num]
    pmol = Chem.MolFromSmiles(products)
    rmol = Chem.MolFromSmiles(reactants)
    patom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in pmol.GetAtoms()}
    ratom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in rmol.GetAtoms()}
    used_atom = set()
    grow_atoms = []
    broken_bonds = []
    changed_bonds = []
    
    # cut bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_break(pbond, rbond): # cut bond
                broken_bonds.append((a, b))
                used_atom.update([a, b])
    
    # Add LG
    for a in edit_num:
        if a in used_atom:
            continue
        patom = pmol.GetAtomWithIdx(patom_map[a])
        ratom = rmol.GetAtomWithIdx(ratom_map[a])
        if check_atom_change(patom, ratom):
            used_atom.update([a])
            grow_atoms.append(a)
            
    # change bond type   
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_change(pbond, rbond):
                if a not in used_atom and b not in used_atom:
                    changed_bonds.append((a, b))
                    changed_bonds.append((b, a))
                    
    used_atoms = set(grow_atoms + [atom for bond in broken_bonds+changed_bonds for atom in bond])
    remote_atoms = [atom for atom in edit_num if atom not in used_atoms]
    remote_atoms_ = []
    for a in remote_atoms:
        atom = rmol.GetAtomWithIdx(ratom_map[a])
        neighbors_map = [n.GetAtomMapNum() for n in atom.GetNeighbors()]
        connected_neighbors = [b for b in used_atoms if b in neighbors_map]
        if len(connected_neighbors) > 0:
            pass
        else:
            for n in neighbors_map:
                remote_atoms_.append(a)
                
    return grow_atoms, broken_bonds, changed_bonds, remote_atoms_

def label_foward_edit_site(reactants, products, edit_num):
    edit_num = [int(num) for num in edit_num]
    rmol = Chem.MolFromSmiles(reactants)
    pmol = Chem.MolFromSmiles(products)
    ratom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in rmol.GetAtoms()}
    patom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in pmol.GetAtoms()}
    atom_symbols = {atom.GetAtomMapNum():atom.GetSymbol() for atom in rmol.GetAtoms()}

    formed_bonds = []
    broken_bonds = []
    changed_bonds = []
    acceptors1 = set()
    acceptors2 = set()
    donors = set()
    form_bond = False
    break_bond = False
    change_bond = False
       
    # cut bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_break(rbond, pbond):
                if a in patom_map:
                    broken_bonds.append((a, b))
                    acceptors1.add(a)
                if b in patom_map:
                    broken_bonds.append((b, a))
                    acceptors1.add(b)
                break_bond = True
                
    # change bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_change(rbond, pbond):
                changed_bonds.append((a, b))
                changed_bonds.append((b, a))
                change_bond = True
                acceptors2.update([a, b])
                
    symmetric = True
    # form bond
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            try:
                pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            except:
                pbond = None
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_formed(rbond, pbond): # cut bond
                form_bond = True
                if a not in acceptors1 and b not in acceptors1 and a not in acceptors2 and b not in acceptors2 :
                    formed_bonds.append((a, b))
                    formed_bonds.append((b, a))
                elif a in acceptors1 and b in acceptors1:
                    symmetric = False
                    formed_bonds.append((a, b))
                    formed_bonds.append((b, a))
                else:
                    symmetric = False
                    if a in acceptors1:
                        formed_bonds.append((b, a))
                    elif a in acceptors2 and b not in acceptors1:
                        formed_bonds.append((b, a))
                    if b in acceptors1:
                        formed_bonds.append((a, b))
                    elif b in acceptors2 and a not in acceptors1:
                        formed_bonds.append((a, b))

    if not symmetric:
        new_changed_bonds = []
        # electron acceptor propagation
        acceptors = set([bond[1] for bond in formed_bonds]).union(acceptors1)
        for atom in acceptors:
            for bond in changed_bonds:
                if bond[0] == atom:
                    new_changed_bonds.append(bond)
        donors = set([bond[0] for bond in formed_bonds])
        for atom in donors:
            for bond in changed_bonds:
                if bond[1] == atom:
                    new_changed_bonds.append(bond)
        changed_bonds = list(set(new_changed_bonds))
        
    used_atoms = set([atom for bond in formed_bonds+broken_bonds+changed_bonds for atom in bond])
    remote_atoms = [atom for atom in edit_num if atom not in used_atoms]
    remote_bonds = []
    for a in remote_atoms:
        atom = rmol.GetAtomWithIdx(ratom_map[a])
        neighbors_map = [n.GetAtomMapNum() for n in atom.GetNeighbors()]
        connected_neighbors = [b for b in used_atoms if b in neighbors_map]
        if len(connected_neighbors) > 0:
            pass
        else:
            for n in neighbors_map:
                remote_bonds.append((a, n))
    return formed_bonds, broken_bonds, changed_bonds, remote_bonds

def label_CHS_change(smiles1, smiles2, edit_num, replacement_dict, use_stereo):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    atom_map_dict1 = {atom.GetAtomMapNum():atom.GetIdx() for atom in mol1.GetAtoms()}
    atom_map_dict2 = {atom.GetAtomMapNum():atom.GetIdx() for atom in mol2.GetAtoms()}
    H_dict = defaultdict(dict)
    C_dict = defaultdict(dict)
    S_dict = defaultdict(dict)
    for atom_map in edit_num:
        atom_map = int(atom_map)
        if atom_map in atom_map_dict2:
            atom1, atom2 = mol1.GetAtomWithIdx(atom_map_dict1[atom_map]), mol2.GetAtomWithIdx(atom_map_dict2[atom_map])
            H_dict[atom_map]['smiles1'], C_dict[atom_map]['smiles1'], S_dict[atom_map]['smiles1'] = atom1.GetNumExplicitHs(), int(atom1.GetFormalCharge()), chiral_type_map[atom1.GetChiralTag()]
            H_dict[atom_map]['smiles2'], C_dict[atom_map]['smiles2'], S_dict[atom_map]['smiles2'] = atom2.GetNumExplicitHs(), int(atom2.GetFormalCharge()), chiral_type_map[atom2.GetChiralTag()]
            
    H_change = {replacement_dict[k]:v['smiles2'] - v['smiles1'] for k, v in H_dict.items()}
    Charge_change = {replacement_dict[k]:v['smiles2'] - v['smiles1'] for k, v in C_dict.items()}
    Chiral_change = {replacement_dict[k]:v['smiles2'] - v['smiles1'] for k, v in S_dict.items()}
    for k, v in S_dict.items():
        if v['smiles2'] == v['smiles1'] or not use_stereo: # no chiral change
            Chiral_change[replacement_dict[k]] = 0
#         elif v['smiles1'] != 0: # opposite the stereo bond
#             Chiral_change[replacement_dict[k]] = 3
        else:
            Chiral_change[replacement_dict[k]] = v['smiles2']
    return atom_map_dict1, H_change, Charge_change, Chiral_change

def bondmap2idx(bond_maps, idx_dict, temp_dict, sort = False, remote = False):
    bond_idxs = [(idx_dict[bond_map[0]], idx_dict[bond_map[1]]) for bond_map in bond_maps]
    if remote:
        bond_temps = list(set([(temp_dict[bond_map[0]], -1) for bond_map in bond_maps]))
        return (bond_idxs, bond_maps, bond_temps)
    else:
        bond_temps = [(temp_dict[bond_map[0]], temp_dict[bond_map[1]]) for bond_map in bond_maps]
    if not sort:
        return (bond_idxs, bond_maps, bond_temps)
    else:
        sort_bond_idxs = []
        sort_bond_maps = []
        sort_bond_temps = []
        for bond1, bond2, bond3 in zip(bond_idxs, bond_maps, bond_temps):
            if bond3[0] < bond3[1]:
                sort_bond_idxs.append(bond1)
                sort_bond_maps.append(bond2)
                sort_bond_temps.append(bond3)
            else:
                sort_bond_idxs.append(tuple(bond1[::-1]))
                sort_bond_maps.append(tuple(bond2[::-1]))
                sort_bond_temps.append(tuple(bond3[::-1]))
        return (sort_bond_idxs, sort_bond_maps, sort_bond_temps)

def atommap2idx(atom_maps, idx_dict, temp_dict):
    atom_idxs = [idx_dict[atom_map] for atom_map in atom_maps]
    atom_temps = [temp_dict[atom_map] for atom_map in atom_maps]
    return (atom_idxs, atom_maps, atom_temps)

def match_label(reactants, products, replacement_dict, edit_num, retro = True, remote = True, use_stereo = True):
    if retro:
        smiles1 = products
        smiles2 = reactants
    else:
        smiles1 = reactants
        smiles2 = products
        
    replacement_dict = {int(k): int(v) for k, v in replacement_dict.items()}
    atom_map_dict, H_change, Charge_change, Chiral_change = label_CHS_change(smiles1, smiles2, edit_num, replacement_dict, use_stereo)
    
    if retro:
        ALG_atoms, broken_bonds, changed_bonds, remote_atoms = label_retro_edit_site(smiles1, smiles2, edit_num)
        edits = {'A': atommap2idx(ALG_atoms, atom_map_dict, replacement_dict),
                  'B': bondmap2idx(broken_bonds, atom_map_dict, replacement_dict, True),
                  'C': bondmap2idx(changed_bonds, atom_map_dict, replacement_dict)}
        if remote:
                  edits['R'] = atommap2idx(remote_atoms, atom_map_dict, replacement_dict)
    else:
        formed_bonds, broken_bonds, changed_bonds, remote_bonds = label_foward_edit_site(smiles1, smiles2, edit_num)
        edits = {'A': bondmap2idx(formed_bonds, atom_map_dict, replacement_dict),
                  'B': bondmap2idx(broken_bonds, atom_map_dict, replacement_dict),
                  'C': bondmap2idx(changed_bonds, atom_map_dict, replacement_dict)}
        if remote:
                  edits['R'] = bondmap2idx(remote_bonds, atom_map_dict, replacement_dict, False, True)
    return edits, H_change, Charge_change, Chiral_change
