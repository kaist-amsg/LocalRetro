from argparse import ArgumentParser

import torch
import sklearn
import torch.nn as nn

from utils import *

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

        loss_a = loss_criterion(atom_logits, atom_labels)
        loss_b = loss_criterion(bond_logits, bond_labels)
        total_loss = torch.cat([loss_a, loss_b]).mean()
        train_loss += total_loss.item()
        
        optimizer.zero_grad()      
        total_loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), args['max_clip'])
        optimizer.step()
                
        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), total_loss), end='', flush=True)

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
            
            loss_a = loss_criterion(atom_logits, atom_labels)
            loss_b = loss_criterion(bond_logits, bond_labels)
            total_loss = torch.cat([loss_a, loss_b]).mean()
            val_loss += total_loss.item()
    return val_loss/batch_id


def main(args):
    args['model_dir'] = '../models/%s/' % args['dataset']
    mkdir_p(args['model_dir'])
    args['model_path'] = args['model_dir'] + 'LocalRetro.pth'
    args['config_path'] = '../data/configs/%s' % args['config']
    args['data_dir'] = '../data/%s' % args['dataset']
    
    args = init_featurizer(args)
    model, loss_criterion, optimizer, scheduler, stopper = load_model(args)
    train_loader, val_loader, test_loader = load_dataloader(args)
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        val_loss = run_an_eval_epoch(args, model, val_loader, loss_criterion)
        print('epoch %d/%d, validation loss: %.4f' %  (epoch + 1, args['num_epochs'], val_loss))
        stopper.filename = args['model_path'].replace('.pth', '_cycle%d.pth' % ((epoch//args['schedule_step'])+1))
        early_stop = stopper.step(1/(epoch+1), model) # no early stop
        scheduler.step()
        
    stopper.load_checkpoint(model)
    test_loss = run_an_eval_epoch(args, model, test_loader, loss_criterion)
    print('test loss: %.4f' % test_loss)
    average_models(args)
    return 

if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='USPTO_50K', help='Dataset to use')
    parser.add_argument('-c', '--config', default='default_config.json', help='Configuration of model')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int, default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    args['mode'] = 'train'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])
    main(args)
