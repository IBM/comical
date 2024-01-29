import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import compute_auc
from src.models import comical, comical_new_emb, comical_new_emb_clasf, mlp_only
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def test_model(config, data=None, subset_index=None, best_checkpoint_name=None):    
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

    return total_loss.item(), acc, data_auc_plot

def calculate_acc(seq,snp_id,idp,idp_id, probs_seq, probs_idp, dd_idp, dd, idp_id_map, snp_id_map):
    correct = 0.0
    for i,j in enumerate(probs_seq):
        if idp_id_map[idp_id[j]] in dd[snp_id_map[snp_id[i]]][0]:
            correct += 1

    for i,j in enumerate(probs_idp):
        if snp_id_map[snp_id[j]] in dd_idp[idp_id_map[idp_id[i]]][0]:
            correct += 1

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