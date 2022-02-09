from collections import defaultdict
import pandas as pd
import errno, sys, os, re
from argparse import ArgumentParser

import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
RDLogger.DisableLog('rdApp.*')

sys.path.append('../')
from LocalTemplate.template_extractor import extract_from_reaction

def mkdir_p(path):
    try:
        os.makedirs(path)
        print('Created directory %s'% path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory %s already exists.' % path)
        else:
            raise
            
def build_template_extractor(args):
    setting = {'verbose': False, 'use_stereo': False, 'use_symbol': False, 'max_unmap': 5, 'retro': False, 'remote': True, 'least_atom_num': 2}
    for k in setting.keys():
        if k in args.keys():
            setting[k] = args[k]
    if args['retro']:
        setting['use_symbol'] = True
    print ('Template extractor setting:', setting)
    return lambda x: extract_from_reaction(x, setting)

def get_reaction_template(extractor, rxn, _id = 0):
    rxn = {'reactants': rxn.split('>>')[0], 'products': rxn.split('>>')[1], '_id': _id}
    result = extractor(rxn)
    return rxn, result

def get_full_template(template, H_change, Charge_change, Chiral_change):
    H_code = ''.join([str(H_change[k+1]) for k in range(len(H_change))])
    Charge_code = ''.join([str(Charge_change[k+1]) for k in range(len(Charge_change))])
    Chiral_code = ''.join([str(Chiral_change[k+1]) for k in range(len(Chiral_change))])
    if Chiral_code == '':
        return '_'.join([template, H_code, Charge_code])
    else:
        return '_'.join([template, H_code, Charge_code, Chiral_code])
            
def extract_templates(args, extractor):
    rxns = pd.read_csv('../data/%s/raw_train.csv' % args['dataset'])['reactants>reagents>production']
    
    class_train = '../data/%s/class_train.csv' % args['dataset']
    if os.path.exists(class_train):
        RXNHASCLASS = True
        rxn_class = pd.read_csv(class_train)['class']
        template_rxnclass = {i+1:set() for i in range(10)}
    else:
        RXNHASCLASS = False
    
    TemplateEdits = {}
    TemplateCs = {}
    TemplateHs = {}
    TemplateSs = {}
    TemplateFreq = defaultdict(int)
    templates_A = defaultdict(int)
    templates_B = defaultdict(int)
    
    for i, reaction in enumerate(rxns):
        if RXNHASCLASS:
            template_class = rxn_class[i]
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            if 'reactants' not in result or 'reaction_smarts' not in result.keys():
                print ('\ntemplate problem: id: %s' % i)
                continue
            reactant = result['reactants']
            template = result['reaction_smarts']
            edits = result['edits']
            H_change = result['H_change']
            Charge_change = result['Charge_change']
            if args['use_stereo']:
                Chiral_change = result['Chiral_change']
            else:
                Chiral_change = {}
            template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
            if template_H not in TemplateHs.keys():
                TemplateEdits[template_H] = {edit_type: edits[edit_type][2] for edit_type in edits}
                TemplateHs[template_H] = H_change
                TemplateCs[template_H] = Charge_change
                TemplateSs[template_H] = Chiral_change

            TemplateFreq[template_H] += 1

            if args['retro']:
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type in ['A', 'R']:
                            templates_A[template_H] += 1
                        else:
                            templates_B[template_H] += 1

            else:
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type != 'A':
                            templates_A['%s_%s' % (template_H, edit_type)] += 1
                        else:
                            templates_B['%s_%s' % (template_H, edit_type)] += 1

            if RXNHASCLASS:
                template_rxnclass[template_class].add(template_H)
                
        except KeyboardInterrupt:
            print('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            print (i, e)
            
        if i % 100 == 0:
            print ('\r i = %s, # of template: %s, # of atom template: %s, # of bond template: %s' % (i, len(TemplateFreq), len(templates_A), len(templates_B)), end='', flush=True)
    print ('\n total # of template: %s' %  len(TemplateFreq))
    
    if args['retro']:
        derived_templates = {'atom':templates_A, 'bond': templates_B}
    else:
        derived_templates = {'real':templates_A, 'virtual': templates_B}
    
    if RXNHASCLASS:
        pd.DataFrame.from_dict(template_rxnclass, orient = 'index').T.to_csv('%s/template_rxnclass.csv' % args['output_dir'], index = None)
        
    TemplateInfos = pd.DataFrame({'Template': k, 'edit_site':TemplateEdits[k], 'change_H': TemplateHs[k], 'change_C': TemplateCs[k], 'change_S': TemplateSs[k], 'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
    TemplateInfos.to_csv('%s/template_infos.csv' % args['output_dir'])
    
    return derived_templates

def export_template(derived_templates, args):
    for k in derived_templates.keys():
        local_templates = derived_templates[k]
        templates = []
        template_class = []
        template_freq = []
        sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
        c = 1
        for t in sorted_tuples:
            templates.append(t[0])
            template_freq.append(t[1])
            template_class.append(c)
            c += 1
        template_dict = {templates[i]:i+1  for i in range(len(templates)) }
        template_df = pd.DataFrame({'Template' : templates, 'Frequency' : template_freq, 'Class': template_class})

        template_df.to_csv('%s/%s_templates.csv' % (args['output_dir'], k))
    return


if __name__ == '__main__':  
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-r', '--retro', default=True,  help='Retrosyntheis or forward synthesis (True for retrosnythesis)')
    parser.add_argument('-v', '--verbose', default=False,  help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  help='Use stereo info in template extraction')
    args = parser.parse_args().__dict__
    args['output_dir'] = '../data/%s' % args['dataset']
    mkdir_p(args['output_dir'])
        
    extractor = build_template_extractor(args)
    derived_templates = extract_templates(args, extractor)
    export_template(derived_templates, args)

    