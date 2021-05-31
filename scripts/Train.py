import os
import numpy as np
import pandas as pd

import torch
import sklearn
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from functools import partial
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from utils import collate_molgraphs, load_LocalRetro, predict
from load_dataset import USPTODataset

MAX_CLIP = 20 

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, atom_labels, bond_labels = batch_data
        if len(smiles) == 1:
            continue
           
        atom_labels, bond_labels = atom_labels.to(args['device']), bond_labels.to(args['device'])
        
        atom_logits, bond_logits, _ = predict(args, model, bg)

        loss_a = loss_criterion(atom_logits, atom_labels).mean()
        loss_b = loss_criterion(bond_logits, bond_labels).mean()
        total_loss = loss_a + loss_b

        train_loss += total_loss.item()
        
        optimizer.zero_grad()      
        total_loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), MAX_CLIP)
        optimizer.step()
                
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, atom loss %.4f, bond loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss_a.item(), loss_b.item()), end='', flush=True)

    print('\nepoch %d/%d, training loss: %.4f' % (epoch + 1, args['num_epochs'], train_loss/batch_id))

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, atom_labels, bond_labels = batch_data
            atom_labels, bond_labels = atom_labels.to(args['device']), bond_labels.to(args['device'])
            atom_logits, bond_logits, _ = predict(args, model, bg)
            
            loss_a = loss_criterion(atom_logits, atom_labels).mean()
            loss_b = loss_criterion(bond_logits, bond_labels).mean()
            total_loss = loss_a + loss_b
            val_loss += total_loss.item()
        
    return val_loss/batch_id


def main(args, exp_config, train_set, val_set, test_set):
    exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
    exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()        
    args['batch_size'] = exp_config['batch_size']

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader_single = DataLoader(dataset=test_set, batch_size=1,
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    model = load_LocalRetro(exp_config)
    model = model.to(args['device'])

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=exp_config['lr'], weight_decay=exp_config['weight_decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
    
    if os.path.exists(args['model_path']):
        user_answer = input('model.pth exists, want to (a) overlap (b) continue from checkpoint (c) make a new model? ')
        if user_answer == 'a':
            stopper = EarlyStopping(mode = 'lower', patience=exp_config['patience'], filename=args['model_path'])
            print ('Overlap exsited model and train a new model...')
        elif user_answer == 'b':
            stopper = EarlyStopping(mode = 'lower', patience=exp_config['patience'], filename=args['model_path'])
            stopper.load_checkpoint(model)
            print ('Train from exsited model checkpoint...')
        elif user_answer == 'c':
            model_name = input('Enter new model name: ')
            args['model_path'] =  '../models/' + model_name + '.pth'
            stopper = EarlyStopping(mode = 'lower', patience=exp_config['patience'], filename=args['model_path'])
            print ('Training a new model %s' % model_name)
    else:
        stopper = EarlyStopping(mode = 'lower', patience=exp_config['patience'], filename=args['model_path'])
    
    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        # Validation and early stop
        val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion)
        early_stop = stopper.step(val_loss, model) 
        scheduler.step()
        
        print('epoch %d/%d, validation loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
        print('epoch %d/%d, Best loss: %.4f' % (epoch + 1, args['num_epochs'], stopper.best_score))

        if early_stop:
            print ('Early stopped!!')
            break

    stopper.load_checkpoint(model)
   
    val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion)
    print('val loss: %.4f' % val_loss)
    test_loss = run_an_eval_epoch(args, model, test_loader, loss_criterion)
    print('test loss: %.4f' % test_loss)
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('LocalRetro training arguements')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-a', '--use-GRA', default= True, help='Model use GRA or not')
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training.')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    
    args = parser.parse_args().__dict__

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    args = init_featurizer(args)
    model_name = '%s.pth' % args['dataset'] if args['use_GRA'] else '%s_noGRA.pth' % args['dataset']

    args['model_path'] = '../models/' + model_name
    mkdir_p('../models')

    exp_config = get_configure()
    exp_config['ALRT_CLASS'] = len(pd.read_csv('../data/%s/atom_templates.csv' % args['dataset']))
    exp_config['BLRT_CLASS'] = len(pd.read_csv('../data/%s/bond_templates.csv' % args['dataset']))
    exp_config['use_GRA'] = args['use_GRA']
    print ('Loaded %s atom templates and %s bond templates' % (exp_config['ALRT_CLASS'], exp_config['BLRT_CLASS']))

    dataset = USPTODataset(args, 
                        smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                        node_featurizer=args['node_featurizer'],
                        edge_featurizer=args['edge_featurizer'])
    train_set, val_set, test_set = split_dataset(args, dataset)
    main(args, exp_config, train_set, val_set, test_set)