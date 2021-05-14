import os, sys, re
import pandas as pd
from collections import defaultdict

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

sys.path.append('../')
from LocalTemplate.template_decoder import *

def main(args):   
    atom_templates = pd.read_csv('../data/%s/atom_templates.csv' % args['dataset'])
    bond_templates = pd.read_csv('../data/%s/bond_templates.csv' % args['dataset'])
    smiles2smarts = pd.read_csv('../data/%s/smiles2smarts.csv' % args['dataset'])
    class_test = '../data/%s/class_test.csv' % args['dataset']
    if os.path.exists(class_test):
        args['rxn_class_given'] = True
        templates_class = pd.read_csv('../data/%s/template_rxnclass.csv' % args['dataset'])
        test_rxn_class = pd.read_csv('../data/%s/class_test.csv' % args['dataset'])['class']
    else:
        args['rxn_class_given'] = False 

    atom_templates = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
    bond_templates = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
    smarts2E = {smiles2smarts['Smarts_template'][i]: eval(smiles2smarts['edit_site'][i]) for i in smiles2smarts.index}
    smarts2H = {smiles2smarts['Smarts_template'][i]: eval(smiles2smarts['change_H'][i]) for i in smiles2smarts.index}
    
    model_type = 'GRA' if args['use_attention'] else 'noGRA'
   
    args['result_path'] = '../results/%s_%s_outputs/raw_prediction.txt' % (args['dataset'], model_type)
    prediction = pd.read_csv(args['result_path'], sep = '\t')
    
    output_path = args['result_path'].replace('raw', 'decoded')
    output_path_class = args['result_path'].replace('raw', 'decoded_class')
    
    with open(output_path, 'w') as f1, open(output_path_class, 'w') as f2:
        for i in prediction.index:
            all_prediction = []
            class_prediction = []
            for K_prediciton in prediction.columns:
                if 'Edit' not in K_prediciton:
                    continue
                rxn = prediction['Reaction'][i]
                products = rxn.split('>>')[1]
                true_reactants = demap(Chem.MolFromSmiles(rxn.split('>>')[0]))
                true_reactants = '.'.join(sorted(true_reactants.split('.')))
                edition = eval(prediction[K_prediciton][i])
                edit_idx = edition[0]
                template_class = edition[1]
                if type(edit_idx) == type(0):
                    template = atom_templates[template_class]
                else:
                    template = bond_templates[template_class]
                template_idx = smarts2E[template]
                H_change = smarts2H[template]
                try:
                    pred_reactants, _, _ = apply_template(products, template, edit_idx, template_idx, H_change)
                except:
                    pred_reactants = []

                all_prediction += [p for p in pred_reactants if p not in all_prediction]
                
                if args['rxn_class_given']:
                    rxn_class = test_rxn_class[i]
                    if template in templates_class[str(rxn_class)].values:
                        class_prediction += [p for p in pred_reactants if p not in class_prediction]
                    if len (class_prediction) >= args['top_k']:
                        break
                        
                elif len (all_prediction) >= args['top_k']:
                    break
                    
            f1.write('\t'.join(all_prediction) + '\n')
            f2.write('\t'.join(class_prediction) + '\n')
            print('\rDecoding LocalRetro predictions %d/%d' % (i, len(prediction)), end='', flush=True)
            
    print ()
       
if __name__ == '__main__':      
    from argparse import ArgumentParser

    parser = ArgumentParser('Local Retrosynthetic Template Prediction')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-a', '--use-attention', default= True, help='Model use GRA or not')
    parser.add_argument('-k', '--top-k', default= 50, help='Number of top predictions')
    args = parser.parse_args().__dict__
    main(args)
        
        
        