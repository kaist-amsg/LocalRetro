import os, sys, re
import pandas as pd
from collections import defaultdict
from argparse import ArgumentParser
sys.path.append('../')
    
import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

from utils import mkdir_p

from LocalTemplate.template_decoder import *

def dearomatic(template):
    for s in ['[c;', '[o;', '[n;', '[s;', '[c@']:
        template = template.replace(s, s.upper())
    return template

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
    
   
    result_name = '%s.txt' % args['dataset'] if args['GRA'] else '%s_noGRA.txt' % args['dataset']
    prediction = pd.read_csv('../outputs/raw_prediction/' + result_name, sep = '\t')
    
    output_path = '../outputs/decoded_prediction/' + result_name
    output_path_class = '../outputs/decoded_prediction_class/' + result_name
    
    with open(output_path, 'w') as f1, open(output_path_class, 'w') as f2:
        for i in prediction.index:
            all_prediction = []
            class_prediction = []
            rxn = prediction['Reaction'][i]
            products = rxn.split('>>')[1]
            idx_map = get_idx_map(products)
            for K_prediciton in prediction.columns:
                if 'Edit' not in K_prediciton:
                    continue
                edition = eval(prediction[K_prediciton][i])
                edit_idx = edition[0]
                template_class = edition[1]
                if type(edit_idx) == type(0):
                    template = atom_templates[template_class]
                    if len(template.split('>>')[0].split('.')) > 1:
                        edit_idx = idx_map[edit_idx]
                else:
                    template = bond_templates[template_class]
                    edit_idx = tuple(edit_idx)
                    if len(template.split('>>')[0].split('.')) > 1:
                        edit_idx = (idx_map[edit_idx[0]], idx_map[edit_idx[1]])
                        
                template_idx = smarts2E[template]
                H_change = smarts2H[template]
                try:
                    pred_reactants, _, _ = apply_template(products, template, edit_idx, template_idx, H_change)
                except Exception as e:
                    # print (e)
                    pred_reactants = []
                    
                if len(pred_reactants) == 0:
                    try:
                        template = dearomatic(template)
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
    parser = ArgumentParser('Decode Prediction')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-k', '--top-k', default= 50, help='Number of top predictions')
    parser.add_argument('-gra', '--GRA', default= True, help='Model use GRA or not')
    args = parser.parse_args().__dict__
    args['GRA'] = False if args['GRA'] == 'False' else True
    mkdir_p('../outputs/decoded_prediction')
    mkdir_p('../outputs/decoded_prediction_class')
    main(args) 
    