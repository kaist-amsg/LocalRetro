import os, sys, re
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from argparse import ArgumentParser
sys.path.append('../')
    
import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions

from utils import mkdir_p
from LocalTemplate.template_decoder import *

def get_k_predictions(test_id, args):
    raw_prediction = args['raw_predictions'][test_id]
    all_prediction = []
    class_prediction = []
    product = raw_prediction[0]
    predictions = raw_prediction[1:]
    for prediction in predictions:
        mol, pred_site, template, template_info, score = read_prediction(product, prediction, args['atom_templates'], args['bond_templates'], args['template_infos'])
        local_template = '>>'.join(['(%s)' % smarts for smarts in template.split('_')[0].split('>>')])
        decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
        try:
            decoded_smiles = decode_localtemplate(mol, pred_site, local_template, template_info)
            if decoded_smiles == None or str((decoded_smiles, score)) in all_prediction:
                continue
        except Exception as e:
#                     print (e)
            continue
        all_prediction.append(str((decoded_smiles, score)))

        if args['rxn_class_given']:
            rxn_class = args['test_rxn_class'][test_id]
            if template in args['templates_class'][str(rxn_class)].values:
                class_prediction.append(str((decoded_smiles, score)))
            if len (class_prediction) >= args['top_k']:
                break

        elif len (all_prediction) >= args['top_k']:
            break
    return (test_id, (all_prediction, class_prediction))

def main(args):   
    atom_templates = pd.read_csv('../data/%s/atom_templates.csv' % args['dataset'])
    bond_templates = pd.read_csv('../data/%s/bond_templates.csv' % args['dataset'])
    template_infos = pd.read_csv('../data/%s/template_infos.csv' % args['dataset'])
    class_test = '../data/%s/class_test.csv' % args['dataset']
    if os.path.exists(class_test):
        args['rxn_class_given'] = True
        args['templates_class'] = pd.read_csv('../data/%s/template_rxnclass.csv' % args['dataset'])
        args['test_rxn_class'] = pd.read_csv(class_test)['class']
    else:
        args['rxn_class_given'] = False 
    args['atom_templates'] = {atom_templates['Class'][i]: atom_templates['Template'][i] for i in atom_templates.index}
    args['bond_templates'] = {bond_templates['Class'][i]: bond_templates['Template'][i] for i in bond_templates.index}
    args['template_infos'] = {template_infos['Template'][i]: {'edit_site': eval(template_infos['edit_site'][i]), 'change_H': eval(template_infos['change_H'][i]), 'change_C': eval(template_infos['change_C'][i]), 'change_S': eval(template_infos['change_S'][i])} for i in template_infos.index}
    
   
    if args['model'] == 'default':
        result_name = 'LocalRetro_%s.txt' % args['dataset']
    else:
        result_name = 'LocalRetro_%s.txt' % args['model']
    
    prediction_file =  '../outputs/raw_prediction/' + result_name
    raw_predictions = {}
    with open(prediction_file, 'r') as f:
        for line in f.readlines():
            seps = line.split('\t')
            if seps[0] == 'Test_id':
                continue
            raw_predictions[int(seps[0])] = seps[1:]
        
    output_path = '../outputs/decoded_prediction/' + result_name
    output_path_class = '../outputs/decoded_prediction_class/' + result_name
    args['raw_predictions'] = raw_predictions
    # multi_processing
    result_dict = {}
    partial_func = partial(get_k_predictions, args = args)
    with multiprocessing.Pool(processes=8) as pool:
        tasks = range(len(raw_predictions))
        for result in tqdm(pool.imap_unordered(partial_func, tasks), total=len(tasks), desc='Decoding LocalRetro predictions'):
            result_dict[result[0]] = result[1]
    
        
    with open(output_path, 'w') as f1, open(output_path_class, 'w') as f2:
        for i in sorted(result_dict.keys()) :
            all_prediction, class_prediction = result_dict[i]
            f1.write('\t'.join([str(i)] + all_prediction) + '\n')
            f2.write('\t'.join([str(i)] + class_prediction) + '\n')
            print('\rDecoding LocalRetro predictions %d/%d' % (i, len(raw_predictions)), end='', flush=True)
    print ()
       
if __name__ == '__main__':      
    parser = ArgumentParser('Decode Prediction')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-m', '--model', default='default', help='Model to use')
    parser.add_argument('-k', '--top-k', default= 50, help='Number of top predictions')
    args = parser.parse_args().__dict__
    mkdir_p('../outputs/decoded_prediction')
    mkdir_p('../outputs/decoded_prediction_class')
    main(args) 
    