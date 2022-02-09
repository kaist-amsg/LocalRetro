import os, sys, re, copy
import pandas as pd

import rdkit 
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

RDLogger.DisableLog('rdApp.*')  

sys.path.append('../')
from Extract_from_train_data import build_template_extractor, get_reaction_template, get_full_template
    

def get_edit_site_retro(smiles):
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
    
def get_edit_site_forward(smiles):
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
    V = []
    for a in A:
        V += [(a,b) for b in A if a != b and (a,b) not in B]
    return V, B

def labeling_dataset(args, split, template_dicts, template_infos, extractor):
    
    if os.path.exists('%s/preprocessed_%s.csv' % (args['output_dir'], split)) and args['force'] == False:
        print ('%s data already preprocessed...loaded data!' % split)
        return pd.read_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    
    rxns = pd.read_csv('../data/%s/raw_%s.csv' % (args['dataset'], split))['reactants>reagents>production']
    reactants = []
    products = []
    reagents = []
    labels = []
    frequency = []
    success = 0
    
    for i, reaction in enumerate(rxns):
        product = reaction.split('>>')[1]
        reagent = ''
        rxn_labels = []
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            template = result['reaction_smarts']
            reactant = result['reactants']
            product = result['products']
            reagent = '.'.join(result['necessary_reagent'])
            edits = {edit_type: edit_bond[0] for edit_type, edit_bond in result['edits'].items()}
            H_change, Charge_change, Chiral_change = result['H_change'], result['Charge_change'], result['Chiral_change']
            if args['use_stereo']:
                Chiral_change = result['Chiral_change']
            else:
                Chiral_change = {}
            template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
            
            if template_H not in template_infos.keys():
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
                continue
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print (i, e)
            reactants.append(reactant)
            products.append(product)
            reagents.append(reagent)
            labels.append(rxn_labels)
            frequency.append(0)
            continue
        
        edit_n = 0
        for edit_type in edits:
            if edit_type == 'C':
                edit_n += len(edits[edit_type])/2
            else:
                edit_n += len(edits[edit_type])
            
            
        if edit_n <= args['max_edit_n']:
            try:
                success += 1
                if args['retro']:
                    atom_sites, bond_sites = get_edit_site_retro(product)
                    for edit_type, edit in edits.items():
                        for e in edit:
                            if edit_type in ['A', 'R']:
                                rxn_labels.append(('a', atom_sites.index(e), template_dicts['atom'][template_H]))
                            else:
                                rxn_labels.append(('b', bond_sites.index(e), template_dicts['bond'][template_H]))
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)       
                    labels.append(rxn_labels)
                    frequency.append(template_infos[template_H]['frequency'])
                else:
                    if len(reagent) != 0:
                        reactant = '%s.%s' % (reactant, reagent)
                    virtual_sites, real_sites = get_edit_site_forward(reactant)
                    for edit_type, bonds in edits.items():
                        for bond in bonds:
                            if edit_type != 'A':
                                rxn_labels.append(('r', real_sites.index(bond), template_dicts['real']['%s_%s' % (template_H, edit_type)]))
                            else:
                                rxn_labels.append(('v', virtual_sites.index(bond), template_dicts['virtual']['%s_%s' % (template_H, edit_type)]))
                    reactants.append(reactant)
                    products.append(reactant)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(template_infos[template_H]['frequency'])
                
            except Exception as e:
                print (i,e)
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
                continue
                
            if i % 100 == 0:
                print ('\r Processing %s %s data..., success %s data (%s/%s)' % (args['dataset'], split, success, i, len(rxns)), end='', flush=True)
        else:
            print ('\nReaction # %s has too many edits (%s)...may be wrong mapping!' % (i, edit_n))
            reactants.append(reactant)
            products.append(product)
            reagents.append(reagent)
            labels.append(rxn_labels)
            frequency.append(0)
            
    print ('\nDerived tempaltes cover %.3f of %s data reactions' % ((success/len(rxns)), split))
    
    df = pd.DataFrame({'Reactants': reactants, 'Products': products, 'Reagents': reagents, 'Labels': labels, 'Frequency': frequency})
    df.to_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    return df

def make_simulate_output(args, split = 'test'):
    df = pd.read_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    with open('%s/simulate_output.txt' % args['output_dir'], 'w') as f:
        f.write('Test_id\tReactant\tProduct\t%s\n' % '\t'.join(['Edit %s\tProba %s' % (i+1, i+1) for i in range(args['max_edit_n'])]))
        for i in df.index:
            labels = []
            for y in eval(df['Labels'][i]):
                if y != 0:
                    labels.append(y)
            if len(labels) == 0:
                lables = [(0, 0)]
#             print (['%s\t%s' % (l, 1.0) for l in labels])
            string_labels = '\t'.join(['%s\t%s' % (l, 1.0) for l in labels])
            f.write('%s\t%s\t%s\t%s\n' % (i, df['Reactants'][i], df['Products'][i], string_labels))
    return 
    
    
def combine_preprocessed_data(train_pre, val_pre, test_pre, args):

    train_valid = train_pre
    val_valid = val_pre
    test_valid = test_pre
    
    train_valid['Split'] = ['train'] * len(train_valid)
    val_valid['Split'] = ['val'] * len(val_valid)
    test_valid['Split'] = ['test'] * len(test_valid)
    all_valid = train_valid.append(val_valid, ignore_index=True)
    all_valid = all_valid.append(test_valid, ignore_index=True)
    all_valid['Mask'] = [int(f>=args['min_template_n']) for f in all_valid['Frequency']]
    print ('Valid data size: %s' % len(all_valid))
    all_valid.to_csv('%s/labeled_data.csv' % args['output_dir'], index = None)
    return

def load_templates(args):
    template_dicts = {}
    if args['retro']:
        keys = ['atom', 'bond']
    else:
        keys = ['real', 'virtual']
        
    for site in keys:
        template_df = pd.read_csv('%s/%s_templates.csv' % (args['output_dir'], site))
        template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index}
        print ('loaded %s %s templates' % (len(template_dict), site))
        template_dicts[site] = template_dict
                                          
    template_infos = pd.read_csv('%s/template_infos.csv' % args['output_dir'])
    template_infos = {t: {'edit_site': eval(e), 'frequency': f} for t, e, f in zip(template_infos['Template'], template_infos['edit_site'], template_infos['Frequency'])}
    print ('loaded total %s templates' % len(template_infos))
    return template_dicts, template_infos
                                          
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-f', '--force', default=False,  help='Force to preprcess the dataset again')
    parser.add_argument('-r', '--retro', default=True,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-m', '--max-edit-n', default=8,  help='Maximum number of edit number')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')
    parser.add_argument('-min', '--min-template-n', type=int, default=1,  help='Minimum of template frequency')
    args = parser.parse_args().__dict__
    args['output_dir'] = '../data/%s' % args['dataset']
        
    template_dicts, template_infos = load_templates(args)
    extractor = build_template_extractor(args)
    test_pre = labeling_dataset(args, 'test', template_dicts, template_infos, extractor)
    make_simulate_output(args)
    val_pre = labeling_dataset(args, 'val', template_dicts, template_infos, extractor)
    train_pre = labeling_dataset(args, 'train', template_dicts, template_infos, extractor)
    
    combine_preprocessed_data(train_pre, val_pre, test_pre, args)
    
        
        
        