import re
import copy
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem

bond_type_map = {'SINGLE': '-', 'DOUBLE': '=', 'TRIPLE': '#', 'AROMATIC': '@'}

def connect_changed_atoms(mol, changed_atoms):
    changed_atoms_idx = [atom.GetIdx() for atom in mol.GetAtoms() if str(atom.GetAtomMapNum()) in changed_atoms]
    while '.' in AllChem.MolFragmentToSmiles(mol, changed_atoms_idx):
        react_smiles = AllChem.MolFragmentToSmiles(mol, changed_atoms_idx)
        frag_nums = {}
        for f, smi in enumerate(react_smiles.split('.')):
            temp_atoms = re.findall(r"\[.*?]", smi)
            frag_nums[f] = [int(atom.split(':')[-1].split(']')[0]) for atom in temp_atoms]

        for f in frag_nums:
            atommaps = frag_nums[f]
            atomidxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() in atommaps]
            frag_nums[f] = atomidxs

        shortest_len = 100
        for f1 in frag_nums:
            for f2 in frag_nums:
                if f1 <= f2:
                    continue
                for a in frag_nums[f1]:
                    for b in frag_nums[f2]:
                        path = Chem.GetShortestPath(mol, a, b)
                        if len(path) < shortest_len:
                            shortest_len = len(path)
                            connect_atoms = path       
        changed_atoms_idx += [idx for idx in connect_atoms if idx not in changed_atoms_idx]
        if shortest_len == 100:
            break
    connect_atoms_map = [str(mol.GetAtomWithIdx(idx).GetAtomMapNum()) for idx in changed_atoms_idx]
    
    return changed_atoms + [m for m in connect_atoms_map if m not in changed_atoms]


def sort_ring_map(temp_order): 
    all_order = []
    temp_len = len(temp_order)
    for i in range(temp_len):
        reorder_idx = [i-temp_len+j for j in range(temp_len)]
        reorder_order = [temp_order[j] for j in reorder_idx]
        all_order.append('-'.join(reorder_order))
    temp_order = temp_order[::-1]
    for i in range(temp_len):
        reorder_idx = [i-temp_len+j for j in range(temp_len)]
        reorder_order = [temp_order[j] for j in reorder_idx]
        all_order.append('-'.join(reorder_order))
    return sorted(all_order)[0].split('-')

def sort_ring_map(temp_order): 
    all_order = []
    temp_len = len(temp_order)
    for i in range(temp_len):
        reorder_idx = [i-temp_len+j for j in range(temp_len)]
        reorder_order = [temp_order[j] for j in reorder_idx]
        all_order.append('-'.join(reorder_order))
    temp_order = temp_order[::-1]
    for i in range(temp_len):
        reorder_idx = [i-temp_len+j for j in range(temp_len)]
        reorder_order = [temp_order[j] for j in reorder_idx]
        all_order.append('-'.join(reorder_order))
    return sorted(all_order)[0].split('-')

def get_temp_order(smarts):
    Is_Ring = False
    temp_atoms = re.findall(r"\[.*?]", smarts)
    temp_order = []
    bond_smarts = smarts
    atom_dict = {}
    for atom in temp_atoms:
        num = atom.split(':')[-1].split(']')[0]
        atom_dict[num] = atom
        temp_order.append(num)
        bond_smarts = bond_smarts.replace(atom, '')
    if '1' in bond_smarts:
        Is_Ring = True
        bond_smarts = bond_smarts.replace('1', '')
    if Is_Ring:
        temp_order_sort = sort_ring_map(temp_order)
    else:
        temp_order_sort = sorted(['-'.join(temp_order), '-'.join(temp_order[::-1])])[0].split('-')
        if temp_order_sort != temp_order:
            bond_smarts = bond_smarts[::-1]
    return temp_order_sort, bond_smarts, atom_dict, Is_Ring

def get_template_bond(temp_order, bond_smarts):
    bond_match = {}
    for n, _ in enumerate(temp_order):
        bond_match[(temp_order[n], temp_order[n-1])] = bond_smarts[n-1]
        bond_match[(temp_order[n-1], temp_order[n])] = bond_smarts[n-1]
    return bond_match   

def sort_product_atom_map(smarts):
    temp_order, bond_smarts, atom_dict, Is_Ring = get_temp_order(smarts)
    bond_match = get_template_bond(temp_order, bond_smarts)
    recon_temp = []
    for i, o in enumerate(temp_order):
        atom_smarts = atom_dict[o]
        if i+1 <= len(temp_order) - 1:
            end = i+1
        else:
            end = 0
        bond = (temp_order[i], temp_order[end])
        bond_smarts = bond_match[bond]
        
        if i == 0 and Is_Ring:
            recon_temp += [atom_smarts, '1', bond_smarts]
        else:
            recon_temp += [atom_smarts, bond_smarts]
    if Is_Ring:
        recon_temp.append('1')
    else:
        recon_temp = recon_temp[:-1]
    return ''.join(recon_temp)

def sort_template(template, replace_dict):
    product = template.split('>>')[0]
    if product.count(':') == 1 or '(' in template:
        return template, replace_dict
    reactant = template.split('>>')[1]
    reorder = False
    for p in reactant.split('.'):
        bond_temp = copy.copy(p)
        temp_atoms = re.findall(r"\[.*?:\d]", p)
        atom_props = set()
        atom_num = []
        for atom in temp_atoms:
            atom_num.append(atom.split(':')[1].split(']')[0])
            atom_props.add(atom.split(':')[0])
            bond_temp = bond_temp.replace(atom, '')
        if len(bond_temp) == len(temp_atoms)-1 and len(atom_props) == 1 and len(temp_atoms) == 2:
            exhange_num = {}
            product_ = product.replace(':%s' % atom_num[0], ':?%s' % atom_num[1])
            product_ = product_.replace(':%s' % atom_num[1], ':?%s' % atom_num[0])
            product_ = product_.replace('?', '')
            exhange_num[str(atom_num[0])] = str(atom_num[1])
            exhange_num[str(atom_num[1])] = str(atom_num[0])
            product = sorted([product, product_])[0]
            if product == product_:
                reorder = True
                break
    if reorder: 
        for atom_map in replace_dict:
            if replace_dict[atom_map] in exhange_num.keys():
                replace_dict[atom_map] = exhange_num[replace_dict[atom_map]]

    return sort_product_atom_map(product) + '>>' + reactant, replace_dict

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

def check_bond_cut(pbond, rbond):
    if pbond == None and rbond != None:
        return False
    elif pbond != None and rbond == None:
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

def check_atom_change(patom, ratom):
    return atom_neighbors(patom) != atom_neighbors(ratom)
    
def label_retro_edit_site(products, reactants, edit_num):
    edit_num = [int(num) for num in edit_num]
    pmol = Chem.MolFromSmiles(products)
    rmol = Chem.MolFromSmiles(reactants)
    
    patom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in pmol.GetAtoms()}
    ratom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in rmol.GetAtoms()}
    edit_bond = []
    used_atom = set()
    cut_bond = False
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_cut(pbond, rbond): # cut bond
                cut_bond = True
                edit_bond.append((a, b))
                used_atom.update([a, b])
    changed_bond = [] 
    change_bond = False
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_change(pbond, rbond):
                if a not in used_atom and b not in used_atom:
                    changed_bond.append((a, b))
                    change_bond = True
                used_atom.update([a, b])
    edit_atom = []
    add_LG = False
    for a in edit_num:
        if a in used_atom:
            continue
        patom = pmol.GetAtomWithIdx(patom_map[a])
        ratom = rmol.GetAtomWithIdx(ratom_map[a])
        if check_atom_change(patom, ratom):
            edit_atom.append(a)
            add_LG = True
            
    if cut_bond:
        return edit_atom + edit_bond
    
    elif change_bond:
        return edit_atom + changed_bond
    
    else:
        return edit_atom

def label_foward_edit_site(reactants, products, edit_num):
    edit_num = [int(num) for num in edit_num]
    rmol = Chem.MolFromSmiles(reactants)
    pmol = Chem.MolFromSmiles(products)
    
    ratom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in rmol.GetAtoms()}
    patom_map = {atom.GetAtomMapNum():atom.GetIdx() for atom in pmol.GetAtoms()}
    
    connect_atom = []
    used_atom = set()
    form_bond = False
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_cut(pbond, rbond): # cut bond
                form_bond = True
                connect_atom += [a, b]
                used_atom.update([a, b])
    
    edit_atom = []
    lose_LG = False
    for a in edit_num:
        if a in used_atom:
            continue
        patom = pmol.GetAtomWithIdx(patom_map[a])
        ratom = rmol.GetAtomWithIdx(ratom_map[a])
        if check_atom_change(patom, ratom):
            edit_atom.append(a)
            lose_LG = True
            
    changed_bond = [] 
    change_bond = False
    for a in edit_num:
        for b in edit_num:
            if a >= b:
                continue
            pbond = pmol.GetBondBetweenAtoms(patom_map[a], patom_map[b])
            rbond = rmol.GetBondBetweenAtoms(ratom_map[a], ratom_map[b])
            if check_bond_change(pbond, rbond):
                if a not in used_atom and b not in used_atom:
                    changed_bond.append((a, b))
                    change_bond = True
                used_atom.update([a, b])
         
    if form_bond:
        return connect_atom
    
    elif lose_LG:
        return edit_atom
    
    else:
        return changed_bond
    
def match_label(products, reactants, replacement_dict, edit_num, retro = True):
    replacement_dict = {int(k): int(replacement_dict[k]) for k in replacement_dict.keys()} # atommap:tempmap
    atom_map_dict = {atom.GetAtomMapNum():atom.GetIdx() for atom in Chem.MolFromSmiles(products).GetAtoms()}
    atom_symbol_dict = {atom.GetAtomMapNum():atom.GetAtomicNum() for atom in Chem.MolFromSmiles(products).GetAtoms()}
    if retro:
        edit_sites = label_retro_edit_site(products, reactants, edit_num)
    else:
        edit_sites = label_foward_edit_site(products, reactants, edit_num)
        
    H_dict = defaultdict(dict)
    for atom in Chem.MolFromSmiles(products).GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if str(atom_map) in edit_num:
            H_dict[atom_map]['product'] = atom.GetNumExplicitHs()
            
    for atom in Chem.MolFromSmiles(reactants).GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if str(atom_map) in edit_num:
            H_dict[atom_map]['reactants'] = atom.GetNumExplicitHs()
            
    H_change = {replacement_dict[k]:v['reactants'] - v['product'] for k, v in H_dict.items()}
   
    edit_idx = [] # num in molecule idx
    edit_map = [] # num in molecule map 
    temp_idx = [] # num in reaction temp
    for site in edit_sites:
        if type(site) == type(1): #Atom edit
            edit_idx.append(atom_map_dict[site])
            edit_map.append(site)
            temp_idx.append(replacement_dict[site])
        else:  #Bond edit
            symbol_map = tuple(atom_symbol_dict[s] for s in site)
            replace_map = tuple(replacement_dict[s] for s in site)
            atom_idx = tuple(atom_map_dict[s] for s in site)
                    
            if list(symbol_map) == sorted(symbol_map) and symbol_map[0] != symbol_map[1]:
                edit_idx.append(atom_idx)
                edit_map.append(site)
                temp_idx.append(replace_map)
            elif symbol_map[0] == symbol_map[1]:
                if list(replace_map) == sorted(replace_map):
                    edit_idx.append(atom_idx)
                    edit_map.append(site)
                    temp_idx.append(replace_map)
                else:
                    edit_idx.append((atom_idx[1], atom_idx[0]))
                    edit_map.append((site[1], site[0]))
                    temp_idx.append((replace_map[1], replace_map[0]))
            else:
                edit_idx.append((atom_idx[1], atom_idx[0]))
                edit_map.append((site[1], site[0]))
                temp_idx.append((replace_map[1], replace_map[0]))
                
    return (edit_idx, edit_map, temp_idx), H_change