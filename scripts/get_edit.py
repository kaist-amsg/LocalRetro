import os
import numpy as np

import time
import torch
import torch.nn as nn

from utils import predict

import dgl

def get_id_template(a, CLASS_NUM):
    CLASS_NUM = CLASS_NUM + 1 # no template
    edit_idx = a//CLASS_NUM
    template = a%CLASS_NUM
    return (edit_idx, template)

def output2edit(out, CLASS_NUM, top_num):
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    output_rank = [r for r in output_rank if get_id_template(r, CLASS_NUM)[1] != 0][:top_num]
    
    selected_edit = [get_id_template(a, CLASS_NUM) for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
     
    return selected_edit, selected_proba
    
def combined_edit(graph, atom_out, bond_out, ATOM_CLASS, BOND_CLASS, top_num):
    edit_id_a, edit_proba_a = output2edit(atom_out, ATOM_CLASS, top_num)
    edit_id_b, edit_proba_b = output2edit(bond_out, BOND_CLASS, top_num)
    atom_pair_list = torch.transpose(graph.adjacency_matrix().coalesce().indices(), 0, 1).numpy()
    edit_id_b = [(list(atom_pair_list[edit_id[0]]), edit_id[1])  for edit_id in edit_id_b]
    edit_id_c = edit_id_a + edit_id_b
    edit_proba_c = edit_proba_a + edit_proba_b
    edit_rank_c = np.flip(np.argsort(edit_proba_c))[:top_num]
    edit_id_c = [edit_id_c[r] for r in edit_rank_c]
    edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]
    
    return edit_id_c, edit_proba_c
    
def get_bg_partition(bg):
    sg = bg.remove_self_loop()
    gs = dgl.unbatch(sg, sg.batch_num_nodes(), (sg.batch_num_edges() - sg.batch_num_nodes()))
    nodes_sep = [0]
    edges_sep = [0]
    for g in gs:
        nodes_sep.append(nodes_sep[-1] + g.num_nodes())
        edges_sep.append(edges_sep[-1] + g.num_edges())
    return gs, nodes_sep[1:], edges_sep[1:]

def write_edits(args, model, test_loader):
    model.eval()
    with open(args['result_path'], 'w') as f:
        f.write('Test_id\tReaction\t%s\n' % '\t'.join(['Edit %s\tProba %s' % (i+1, i+1) for i in range(args['top_num'])]))
        with torch.no_grad():
            for batch_id, data in enumerate(test_loader):
                _, bg, rxns = data
                batch_atom_logits, batch_bond_logits, _ = predict(args, model, bg)    
                batch_atom_logits = nn.Softmax(dim=1)(batch_atom_logits)
                batch_bond_logits = nn.Softmax(dim=1)(batch_bond_logits) 
                graphs, nodes_sep, edges_sep = get_bg_partition(bg)

                start_node = 0
                start_edge = 0
                print('\rWriting test molecule batch %s/%s' % (batch_id, len(test_loader)), end='', flush=True)
                for single_id, (graph, end_node, end_edge) in enumerate(zip(graphs, nodes_sep, edges_sep)):
                    rxn = rxns[single_id]
                    test_id = (batch_id * args['batch_size']) + single_id
                    edit_id, edit_proba = combined_edit(graph, batch_atom_logits[start_node:end_node], batch_bond_logits[start_edge:end_edge], args['AtomTemplate_n'], args['BondTemplate_n'], args['top_num'])
                    start_node = end_node
                    start_edge = end_edge
                    f.write('%s\t%s\t%s\n' % (test_id, rxn, '\t'.join(['%s\t%.3f' % (edit_id[i], edit_proba[i]) for i in range(args['top_num'])])))

    print ()
    return 