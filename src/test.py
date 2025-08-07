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
import statsmodels.api as sm

import datetime
import json

master_pair_freq_dict = {}

def test_model(config, args, data=None, subset_index=None, best_checkpoint_name=None):    
    # Call functions from dataset to get pair buckets for evaluation of model (accuracy calculation)
    if config['out_flag'] == 'pairs':
        dd_idp, dd = data.get_pairs_for_test()
        idp_id_map, snp_id_map = data.get_token_maps()

    # Define test loader
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,subset_index), batch_size=int(config['batch_size']), shuffle=False)

    # Define model   
    model = comical_new_emb(config) if config['out_flag']=='pairs' else mlp_only(config) if config['out_flag']=='mlp' else comical_new_emb_clasf(config)

    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if config['tune'] == False:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model) # TODO: implement DDP for better paralelization (if needed)
    model = model.to(device)

    # Load state for best model
    if config['tune']:
        # model_state, optimizer_state = torch.load(best_checkpoint_name)
        model_dict = torch.load(best_checkpoint_name)
        model_state = model_dict['model_state']
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
    soft_logits_seq_l = []
    soft_logits_idp_l = []
    data_auc_plot = {'preds':[],'target':[]}
    gen_emb_l, idp_emb_l, cov_l = [], [], []
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
            target = None if config['out_flag'] == 'pairs' else target.to(device).float()
            covariates = None if config['out_flag'] == 'pairs' else covariates.to(device).float()

            if config['save_embeddings']:
                if config['out_flag'] == 'pairs':
                    logits_seq, logits_idp, _,_  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
                else:
                    pred, gen_emb,idp_emb = model(seq,snp_id,idp,idp_id,config['save_embeddings'],covariates)
                    gen_emb_l.append(gen_emb)
                    idp_emb_l.append(idp_emb)
                    cov_l.append(covariates.detach().cpu().numpy())
            else:
                if config['out_flag'] == 'pairs':
                    logits_seq, logits_idp  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
                else:
                    pred  = model(seq,snp_id,idp,idp_id,config['save_embeddings'],covariates)
            
            if config['out_flag'] == 'pairs':
                # Set ground truth following CLIP implementation - output should match batch indeces
                ground_truth = torch.arange(len(seq),dtype=torch.long,device=device)
                total_loss += (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
            else:
                # total_loss += loss_clf(pred,target.long()) if config['out_flag'] == 'clf' else loss_reg(pred,target)
                total_loss = 0.01 # FIXME: temporary fix to avoid loss calculation for now, remove later
            
            # Inputs and Outputs per batch for returning
            seq_l.extend(seq.detach().cpu().numpy())
            idp_l.extend(idp.detach().cpu().numpy())

            if config['out_flag'] == 'pairs':
                # Compute softmax probs to use for accuracy calculation
                if config['decile'] == 0:
                    probs_seq = np.argmax(logits_seq.softmax(dim=-1).cpu().numpy(), axis = 1)
                    probs_idp = np.argmax(logits_idp.softmax(dim=-1).cpu().numpy(), axis = 1)
                else:
                    # New accuracy calculation - overlap between top 10% of predicted values and true values rather than just the top predicted value
                    probs_seq = logits_seq.softmax(dim=-1).cpu().numpy()
                    probs_idp = logits_idp.softmax(dim=-1).cpu().numpy()
                
                # Calculte accuracy per batch
                if config['decile'] == 0:
                    acc += calculate_acc(seq.cpu().numpy(),snp_id.cpu().numpy(),idp.cpu().numpy(),idp_id.cpu().numpy(), probs_seq, probs_idp,dd_idp, dd, idp_id_map, snp_id_map)
                else:
                    acc += calculate_decile_acc(seq.cpu().numpy(),snp_id.cpu().numpy(),idp.cpu().numpy(),idp_id.cpu().numpy(), probs_seq, probs_idp,dd_idp, dd, idp_id_map, snp_id_map, config['decile'])
                # Save softmaxed logits for later analysis
                soft_logits_idp_l.extend(probs_idp) 
                soft_logits_seq_l.extend(probs_seq)
            elif config['out_flag'] == 'clf' or config['out_flag'] == 'mlp':
                # Compute softmax probs to use for accuracy calculation
                probs = np.argmax(pred.softmax(dim=-1).cpu().numpy(), axis = 1)
                # Calculte accuracy per batch
                # acc += np.sum(probs == target.cpu().numpy()) / len(probs)
                # FIXME: Add a column with zeros to pred to match shape - delete later! just to run old code and obtain emebdddings
                pred = torch.cat((pred, torch.ones((pred.shape[0],1), device=device)), dim=1)

                data_auc_plot['preds'].extend(pred.softmax(dim=-1).cpu().numpy()[:,1])
                data_auc_plot['target'].extend(target.cpu().numpy())
                acc += compute_auc(pred.softmax(dim=-1).cpu().numpy()[:,1],target.cpu().numpy()) #substitue AUC for accuracy
            elif config['out_flag'] == 'reg':
                # Calculte R2 per batch using statsmodels OLS
                pred_prs = pred.detach().cpu().numpy()
                pred_prs = np.concatenate((pred_prs, covariates.cpu().numpy()), axis=1)
                pred_prs = sm.add_constant(pred_prs)
                sm_model = sm.OLS(target.cpu().numpy(),pred_prs)
                sm_results = sm_model.fit()
                r2_adj = sm_results.rsquared_adj
                r2 = sm_results.rsquared
                acc += r2
                data_auc_plot['preds'].extend(np.concatenate((pred_prs, covariates.cpu().numpy()), axis=1))
                data_auc_plot['target'].extend(target.cpu().numpy())
            else:
                # Calculte MSE per batch
                acc += F.mse_loss(pred.squeeze(-1),target).item()

    # Full test set metric computation
    total_loss /= (batch_idx+1)
    acc /= (batch_idx+1)

    np.concatenate(gen_emb_l).dump('gen_emb_'+str(args['top_n_perc'])+'.npy')
    np.concatenate(idp_emb_l).dump('idp_emb_'+str(args['top_n_perc'])+'.npy')
    np.concatenate(cov_l)[:,:2].dump('cov_'+str(args['top_n_perc'])+'.npy')

    # if config['out_flag'] == 'pairs':
        # max_length = max(len(lst) for lst in master_pair_freq_dict.values())

        # # Fill shorter lists with zeros
        # for key, value in master_pair_freq_dict.items():
        #     if len(value) < max_length:
        #         master_pair_freq_dict[key] += [0] * (max_length - len(value))

        # master_pair_freq_pd = pd.DataFrame.from_dict(master_pair_freq_dict, orient='index')
        
        # chi2_stat, p_value = chisquare(f_obs = master_pair_freq_pd.values, 
        #                             axis = 1, ddof = master_pair_freq_pd.shape[0] - 1)

        # adj_chi2_stat = (chi2_stat / master_pair_freq_pd.shape[0]).reshape((master_pair_freq_pd.shape[0],1))
        
        # adj_p_value = np.float64(stats.f.sf(adj_chi2_stat, dfn = 1, dfd = master_pair_freq_pd.shape[0] - 1)[:,0])

        # master_pair_freq_pd = master_pair_freq_pd.reset_index() 
        # master_pair_freq_pd['chi'] = chi2_stat
        # master_pair_freq_pd['p'] = p_value
        # master_pair_freq_pd['adj-chi'] = adj_chi2_stat
        # master_pair_freq_pd['adj-p'] = adj_p_value
        # master_pair_freq_pd.rename(columns={'index': 'pair', 'values': 'freq'}, inplace=True)

        # master_pair_freq_pd.to_csv('pairs_freqs_pvals_'+str(args['top_n_perc'])+'.csv', index=False, mode='w')

    return total_loss.item(), acc, data_auc_plot, soft_logits_idp_l, soft_logits_seq_l 

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

def calculate_decile_acc(seq, snp_id, idp, idp_id, probs_seq, probs_idp, dd_idp, dd, idp_id_map, snp_id_map, decile):
    # global master_pair_freq_dict
    # pairs = []
    acc = 0.0
    total_count = len(probs_seq) + len(probs_idp)
    
    for i, prob_seq in enumerate(probs_seq):
        # Calculate the top decile of the predicted values
        top_decile_indices = np.where(pd.qcut(prob_seq, 10, labels=False, duplicates='drop') == decile)[0] #top decile is 9, include all = 1 
        top_decile_idps = {idp_id_map[idp_id[idx]] for idx in top_decile_indices}

        # Obtain the true associated values for the SNP
        true_values = set(dd[snp_id_map[snp_id[i]]][0])

        # Calculate the overlap
        overlap = len(true_values.intersection(top_decile_idps))
        acc += overlap / len(true_values) if true_values else 0

        # pairs.append( (dd_idp[idp_id_map[idp_id[i]]][0][0], idp_id_map[idp_id[j]]) )

    for i, prob_idp in enumerate(probs_idp):
        # Calculate the top decile of the predicted values
        top_decile_indices = np.where(pd.qcut(prob_idp, 10, labels=False, duplicates='drop') == decile)[0]
        top_decile_snps = {snp_id_map[snp_id[idx]] for idx in top_decile_indices}

        # Obtain the true associated values for the IDP
        true_values = set(dd_idp[idp_id_map[idp_id[i]]][0])

        # Calculate the overlap
        overlap = len(true_values.intersection(top_decile_snps))
        acc += overlap / len(true_values) if true_values else 0

        # pairs.append( (snp_id_map[snp_id[j]], dd[snp_id_map[snp_id[i]]][0][0]) )

    # counter = dict(Counter(pairs))

    # for key, value in counter.items():
    #     if key not in master_pair_freq_dict:
    #         master_pair_freq_dict[key] = []
    #         master_pair_freq_dict[key].append(value)
    #     else:
    #         master_pair_freq_dict[key].append(value)

    return acc / total_count / 2

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