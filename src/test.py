import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import compute_auc
from src.models import comical, comical_new_emb, comical_new_emb_clasf, mlp_only
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from collections import Counter
from scipy.stats import chisquare
import scipy.stats as stats

import datetime
import json

master_pair_freq_dict = {}

def test_model(config, args, data=None, subset_index=None, best_checkpoint_name=None):    
    # Call functions from dataset to get pair buckets for evaluation of model (accuracy calculation)
    if config['out_flag'] == 'seq_idp':
        dd_idp, dd = data.get_pairs_for_test()
        idp_id_map, snp_id_map = data.get_token_maps()

    # Define test loader
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,subset_index), batch_size=int(config['batch_size']), shuffle=True)

    # Define model   
    model = comical_new_emb(config) if config['out_flag']=='seq_idp' else comical_new_emb_clasf(config)
    model = mlp_only(config) if config['out_flag']=='mlp' else model

    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model) # TODO: implement DDP for better paralelization (if needed)
    model = model.to(device)

    # Load state for best model
    if config['tune']:
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_name, ".pt"))

    else:
        # model_state, optimizer_state = torch.load(best_checkpoint_name)
        model_dict = torch.load(best_checkpoint_name)
        model_state, optimizer_state = model_dict['model_state_dict'], model_dict['optimizer_state_dict']
    
    model.load_state_dict(model_state)

    # Run testing
    model.eval()
    total_loss = 0.
    acc = 0.
    seq_l = []
    idp_l = []
    data_auc_plot = {'preds':[],'target':[]}
    # Added classification and regression losses - Added 2024.01.10 - Yet to be debugged
    if config['out_flag'] == 'clf' or config['out_flag'] == 'mlp':
        loss_clf = nn.CrossEntropyLoss()
    elif config['out_flag'] == 'reg':
        loss_reg = nn.MSELoss()
    else:
        loss_seq = nn.CrossEntropyLoss()
        loss_idp = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (snp_id,seq,idp_id,idp,target,covariates) in enumerate(tqdm(test_loader,desc='Testing batch loop')):
            seq,snp_id,idp,idp_id = seq.to(device).long(),snp_id.to(device).long(),idp.to(device).long(),idp_id.to(device).long()
            target = None if config['out_flag'] == 'seq_idp' else target.to(device).float()
            covariates = None if config['out_flag'] == 'seq_idp' else covariates.to(device).float()

            if config['save_embeddings']:
                logits_seq, logits_idp, _,_  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
            else:
                if config['out_flag'] == 'seq_idp':
                    logits_seq, logits_idp  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
                else:
                    pred  = model(seq,snp_id,idp,idp_id,config['save_embeddings'],covariates)
            
            if config['out_flag'] == 'seq_idp':
                # Set ground truth following CLIP implementation - output should match batch indeces
                ground_truth = torch.arange(len(seq),dtype=torch.long,device=device)
                total_loss += (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
            else:
                # total_loss += loss_clf(pred,target.long()) if config['out_flag'] == 'clf' else loss_reg(pred,target)
                total_loss += loss_clf(pred,target.long()) 
            
            # Inputs and Outputs per batch for returning
            seq_l.extend(seq.detach().cpu().numpy())
            idp_l.extend(idp.detach().cpu().numpy())

            if config['out_flag'] == 'seq_idp':
                # Compute softmax probs to use for accuracy calculation
                probs_seq = np.argmax(logits_seq.softmax(dim=-1).cpu().numpy(), axis = 1)
                probs_idp = np.argmax(logits_idp.softmax(dim=-1).cpu().numpy(), axis = 1)
                
                # Calculte accuracy per batch
                acc += calculate_acc(seq.cpu().numpy(),snp_id.cpu().numpy(),idp.cpu().numpy(),idp_id.cpu().numpy(), probs_seq, probs_idp,dd_idp, dd, idp_id_map, snp_id_map)
            elif config['out_flag'] == 'clf' or config['out_flag'] == 'mlp':
                # Compute softmax probs to use for accuracy calculation
                probs = np.argmax(pred.softmax(dim=-1).cpu().numpy(), axis = 1)
                # Calculte accuracy per batch
                # acc += np.sum(probs == target.cpu().numpy()) / len(probs)
                data_auc_plot['preds'].extend(pred.softmax(dim=-1).cpu().numpy()[:,1])
                data_auc_plot['target'].extend(target.cpu().numpy())
                acc += compute_auc(pred.softmax(dim=-1).cpu().numpy()[:,1],target.cpu().numpy()) #substitue AUC for accuracy
            else:
                # Calculte MSE per batch
                acc += F.mse_loss(pred,target).item()

    # Full test set metric computation
    total_loss /= (batch_idx+1)
    acc /= (batch_idx+1)

    max_length = max(len(lst) for lst in master_pair_freq_dict.values())

    # Fill shorter lists with zeros
    for key, value in master_pair_freq_dict.items():
        if len(value) < max_length:
            master_pair_freq_dict[key] += [0] * (max_length - len(value))

    master_pair_freq_pd = pd.DataFrame.from_dict(master_pair_freq_dict, orient='index')
    
    chi2_stat, p_value = chisquare(f_obs = master_pair_freq_pd.values, 
                                   axis = 1, ddof = master_pair_freq_pd.shape[0] - 1)

    adj_chi2_stat = (chi2_stat / master_pair_freq_pd.shape[0]).reshape((master_pair_freq_pd.shape[0],1))
    
    adj_p_value = np.float64(stats.f.sf(adj_chi2_stat, dfn = 1, dfd = master_pair_freq_pd.shape[0] - 1)[:,0])

    master_pair_freq_pd = master_pair_freq_pd.reset_index() 
    master_pair_freq_pd['chi'] = chi2_stat
    master_pair_freq_pd['p'] = p_value
    master_pair_freq_pd['adj-chi'] = adj_chi2_stat
    master_pair_freq_pd['adj-p'] = adj_p_value
    master_pair_freq_pd.rename(columns={'index': 'pair', 'values': 'freq'}, inplace=True)

    master_pair_freq_pd.to_csv('pairs_freqs_pvals_'+str(args['top_n_perc'])+'.csv', index=False, mode='w')

    return total_loss.item(), acc, data_auc_plot 

def calculate_acc(seq,snp_id,idp,idp_id, probs_seq, probs_idp, dd_idp, dd, idp_id_map, snp_id_map):
    global master_pair_freq_dict
    pairs = []
    correct = 0.0
    for i,j in enumerate(probs_seq):
        if idp_id_map[idp_id[j]] in dd[snp_id_map[snp_id[i]]][0]:
            correct += 1
            pairs.append( (dd_idp[idp_id_map[idp_id[i]]][0][0], idp_id_map[idp_id[j]]) ) # flipped from Diego order so its always (SNP, IDP)

    for i,j in enumerate(probs_idp):
        if snp_id_map[snp_id[j]] in dd_idp[idp_id_map[idp_id[i]]][0]:
            correct += 1
            pairs.append( (snp_id_map[snp_id[j]], dd[snp_id_map[snp_id[i]]][0][0]) )

    counter = dict(Counter(pairs))

    for key, value in counter.items():
        if key not in master_pair_freq_dict:
            master_pair_freq_dict[key] = []
            master_pair_freq_dict[key].append(value)
        else:
            master_pair_freq_dict[key].append(value)

    acc = correct / len(probs_seq) /2 
    return acc

def compute_auc(output,target):
    # FIXME: currently if batch has only one class return AUC of 0.5
    if len(np.unique(target)) == 1:
        return 0.5
    
    if len(np.unique(target)) == 2:
        micro_roc_auc_ovo = roc_auc_score(
            target,
            output
        )
    else:
        micro_roc_auc_ovo = roc_auc_score(
            target,
            output,
            multi_class="ovo",
            average="macro",
            labels=np.asarray([0,1,2])
        )

    return micro_roc_auc_ovo