import re
from rdkit import Chem
from .template_extractor import *

def get_temp_order(smarts, ring_tag = None):
    temp_atoms = re.findall(r"\[.*?]", smarts)
    temp_order = []
    bond_smarts = smarts
    atom_dict = {}
    for atom in temp_atoms:
        num = atom.split(':')[-1].split(']')[0]
        atom_dict[num] = atom
        if not ring_tag:
            temp_order.append(num)
            bond_smarts = bond_smarts.replace(atom, '')
        elif num in ring_tag:
            temp_order.append(num)
            bond_smarts = bond_smarts.replace(atom, '')
        else:
            bond_smarts = bond_smarts.replace(atom, '')
            bond_smarts = bond_smarts.replace(bond_smarts[0], '')
    bond_smarts = bond_smarts.replace('1', '')   
    return temp_order, bond_smarts, atom_dict

def enumerate_to_match(order1, order2, C = None, second_trial = False):
    start = order2[0]
    end = order2[-1]
    if C and not second_trial:
        C_num = get_temp_order(C)[0]
        for c in C_num:
            order1.remove(c)    
    temp_len = len(order1)
    get_order = False
    for i in range(temp_len):
        reorder_idx = [i-temp_len+j for j in range(temp_len)]
        reorder_order = [order1[j] for j in reorder_idx]
        if reorder_order[0] == start and reorder_order[-1] == end:
            get_order = True
            break
            
    if not get_order and not second_trial:
        reorder_order = enumerate_to_match(order1[::-1], order2, C, True)
    return reorder_order

def get_ABC_index(temp_order, frag_order):
    frag_len = len(temp_order) - sum([len(f) - 1 for f in frag_order])
    ABC_index = []
    hidden_index = []
    for frag in frag_order:
        match_idx = []
        for f in frag:
            match_idx.append(''.join(temp_order).index(f))
            
        smallest_idx = sorted(match_idx)[0]
        hidden_index += [idx for idx in match_idx if idx != smallest_idx]
        ABC_index.append(smallest_idx)
    
    for i, idx in enumerate(ABC_index):
        reduce_num = sum([idx > hidx for hidx in hidden_index])
        ABC_index[i] = idx - reduce_num
        
    for i in range(len(temp_order)):
        n = [(a + i) % frag_len for a in ABC_index]
        if max(n) < len(frag_order):
            ABC_index = n
            break    
    ABC_order = [ABC_index.index(i) for i in range(len(frag_order))]
    return ABC_order

def find_cut_bond(temp_order, frag_order):
    extend_order = temp_order + [temp_order[0]]
    temp_round = '-'.join(extend_order)
    for atom1 in frag_order[0]:
        for atom2 in frag_order[1]:
            bond = atom1 + '-' + atom2
            if bond in temp_round or bond in temp_round[::-1]:
                return bond.split('-')
            else:
                pass
    return None

def reverse_lg(lg):
    symbol_string = ''
    string_order = 0
    New_Alpha = True
    inverse = True
    meet_alpha = False
    temp = ''
    elements = defaultdict(str)
    reverse_lg = lg[::-1]
    for i, char in enumerate(reverse_lg):    
        if inverse and char.isalpha():
            if New_Alpha: # last word is alpha  
                string_order += 1 
                symbol_string += str(string_order) + '*'
                elements[str(string_order)] += char
                New_Alpha = False  
            else:
                elements[str(string_order)] += char 
                New_Alpha = False  
              
        elif char == ')':
            inverse = False
            temp += char
            New_Alpha = True
            
        elif char == '(':
            if reverse_lg[i+1] == ')':
                temp += char
            else:
                meet_alpha = True
                temp += char
  
        elif inverse:
            symbol_string += char
            New_Alpha = True
            
        elif meet_alpha:
            temp += char
            symbol_string += temp[::-1]
            temp = ''
            meet_alpha = False
            inverse = True

        else:
            temp += char 
            
    for i in range(string_order):
        e = str(i+1)
        symbol_string = symbol_string.replace(e + '*', elements[e][::-1])
        
    return symbol_string
    
def get_ABC(temp_order, smarts):
    split_smarts = smarts.split('.')
    atom_dict = get_temp_order(smarts)[2]
    frag_order = [get_temp_order(smarts)[0] for smarts in split_smarts]
    if smarts.count('.') == 2:
        ABC_order = get_ABC_index(temp_order, frag_order)
        C = split_smarts[ABC_order[1]]
        AB = frag_order[ABC_order[0]] + frag_order[ABC_order[-1]]
    else:
        ABC_order = [0, 1]
        C = None
        AB = find_cut_bond(temp_order, frag_order)

    a = split_smarts[ABC_order[0]].split('[')[0]
    b = split_smarts[ABC_order[-1]].split('[')[0]
    if len(a) > 1:  
        atom_dict[AB[0]] = atom_dict[AB[0]] + '(%s)' % reverse_lg(a)
    if len(b) > 1:
        atom_dict[AB[-1]] = atom_dict[AB[-1]] + '(%s)' % reverse_lg(b)

    return AB, C, atom_dict

def get_template_bond1(temp_order, bond_dict):
    bond_match = {}
    for n, _ in enumerate(temp_order):
        start = temp_order[n-1]
        end = temp_order[n]
        bond_match[(end, start)] = bond_dict[n-1]
        bond_match[(start, end)] = bond_dict[n-1]
    return bond_match       
    
def get_between_strings(string, start, end):
    return string[string.find(start)+len(start):string.rfind(end)]

def get_template_bond2(smarts):
    bond_match = {}
    for frag in smarts.split('.'):
        temp_atoms = re.findall(r"\[.*?]", frag)
        atom_order = []
        if len(temp_atoms) == 1:
            continue
        bond_smarts = frag
        for atom in temp_atoms:
            num = atom.split(':')[-1].split(']')[0]
            atom_order.append(num)
            bond_smarts = bond_smarts.replace(atom, num)
            
        for n, _ in enumerate(atom_order):
            start = atom_order[n-1]
            end = atom_order[n]
            bond_match[(start, end)] = get_between_strings(bond_smarts, start, end)
            bond_match[(end, start)] = get_between_strings(bond_smarts, start, end)
    return bond_match

def reconstruct_smarts(reorder_order, bond_match1, bond_match2, atom_dict1, atom_dict2):
    recon_temp = []
    used_atom = []
    for i, o in enumerate(reorder_order):
        if o in atom_dict2.keys():
            atom_smarts = atom_dict2[o]
        else:
            atom_smarts = atom_dict1[o]
        
        used_atom.append(o)
        bond = (reorder_order[i-1], reorder_order[i])
        if bond in bond_match2.keys():
            bond_smarts = bond_match2[bond]
        else:
            bond_smarts = bond_match1[bond]
        if i == 0:
            recon_temp.append(atom_smarts)
        else:
            recon_temp += [bond_smarts, atom_smarts]
            
    for atom in atom_dict2.keys():
        if atom not in used_atom:
            atom_smarts = atom_dict2[atom]
            bond1 = (reorder_order[0], atom)
            bond2 = (reorder_order[-1], atom)
            if bond1 in bond_match2.keys():
                bond_smarts = bond_match2[bond1]
                recon_temp = [atom_smarts, bond_smarts] + recon_temp
            if bond2 in bond_match2.keys():
                bond_smarts = bond_match2[bond2]
                recon_temp += [bond_smarts, atom_smarts]
        
    return ''.join(recon_temp)

def reconstruct_template(smarts1, smarts2, ring_tag):
    temp_order, bond_dict1, atom_dict1 = get_temp_order(smarts1, ring_tag)
    AB_order, C, atom_dict2 = get_ABC(temp_order, smarts2)  
    reorder_order = enumerate_to_match(temp_order, AB_order, C)
    bond_match1 = get_template_bond1(temp_order, bond_dict1)
    bond_match2 = get_template_bond2(smarts2)
    recon_temp1 = smarts1
    recon_temp2 = reconstruct_smarts(reorder_order, bond_match1, bond_match2, atom_dict1, atom_dict2)
    if C:
        recon_temp2 = C + '.' + recon_temp2
    return recon_temp1 + '>>' + recon_temp2

def include_ring_info(smiles, template, edit_idx, temp_idx, matched_idx):
    mol = Chem.MolFromSmiles(smiles)
    if type(edit_idx) == type(0) or '.' not in template or template.split('>>')[0][-1] == '1':
        return template   
    elif ']-O-[C' in template or (len(temp_idx) == 2 and template.count('.') == 1):
        return template
    else:
        edit_bond = mol.GetBondBetweenAtoms(edit_idx[0], edit_idx[1])
        if not edit_bond.IsInRing():
            return template
        else:
            ri = mol.GetRingInfo()
            for i, ring in enumerate(ri.BondRings()):
                if edit_bond.GetIdx() in ring:
                    ring_atoms = ri.AtomRings()[i]
    
    changed_atom_idx =  [idx for idx in matched_idx.keys()]
    changed_atom_idx += [idx for idx in ring_atoms if idx not in changed_atom_idx]
    
    ring_atoms_map = {}
    ring_tag = []
    ummap_num = len(matched_idx) + 1
    for k in changed_atom_idx:
        if k in matched_idx.keys():
            ring_atoms_map[k] = matched_idx[k]
        else:
            ring_atoms_map[k]  = ummap_num
            ummap_num += 1
        if k in ring_atoms:
            ring_tag.append(str(ring_atoms_map[k]))
        
    for atom in mol.GetAtoms():
        if atom.GetIdx() in changed_atom_idx:
            atom.SetAtomMapNum(ring_atoms_map[atom.GetIdx()])
        else:
            atom.SetAtomMapNum(0)

    
    changed_atom_tags = [str(atom.GetAtomMapNum()) for atom in mol.GetAtoms() if atom.GetIdx() in changed_atom_idx]
        
    mols = mols_from_smiles_list(replace_deuterated(Chem.MolToSmiles(mol)).split('.'))
    reactant_fragments, intra_only, dimer_only = get_fragments_for_changed_atoms(mols, changed_atom_tags, 
                radius = 0, expansion = [], category = 'product', local = True)
    
    temp1 = ''.join(canonicalize_template(reactant_fragments))[1:-1]
    temp2 = template.split('>>')[1]
    return reconstruct_template(temp1, temp2, ring_tag)
