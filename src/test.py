import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import compute_auc
from src.models import comical, comical_new_emb
import numpy as np
from tqdm import tqdm


def test_model(config, data=None, subset_index=None, best_checkpoint_name=None):    
    # Call functions from dataset to get pair buckets for evaluation of model (accuracy calculation)
    dd_idp, dd = data.get_pairs_for_test()
    idp_id_map, snp_id_map = data.get_token_maps()

    # Define test loader
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,subset_index), batch_size=int(config['batch_size']), shuffle=True)

    # Define model   
    model = comical_new_emb(config)

    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model) # TODO: implement DDP for better paralelization (if needed)
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
    loss_seq = nn.CrossEntropyLoss()
    loss_idp = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (snp_id,seq,idp_id,idp) in enumerate(tqdm(test_loader,desc='Testing batch loop')):
            seq,snp_id,idp,idp_id = seq.to(device).long(),snp_id.to(device).long(),idp.to(device).long(),idp_id.to(device).long()

            if config['save_embeddings']:
                logits_seq, logits_idp, _,_  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
            else:
                logits_seq, logits_idp  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
            # Set ground truth following CLIP implementation - output should match batch indeces
            ground_truth = torch.arange(len(seq),dtype=torch.long,device=device)
            total_loss += (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2

            # Inputs and Outputs per batch for returning
            seq_l.extend(seq.detach().cpu().numpy())
            idp_l.extend(idp.detach().cpu().numpy())

            # Compute softmax probs to use for accuracy calculation
            probs_seq = np.argmax(logits_seq.softmax(dim=-1).cpu().numpy(), axis = 1)
            probs_idp = np.argmax(logits_idp.softmax(dim=-1).cpu().numpy(), axis = 1)
            
            # Calculte accuracy per batch
            acc += calculate_acc(seq.cpu().numpy(),snp_id.cpu().numpy(),idp.cpu().numpy(),idp_id.cpu().numpy(), probs_seq, probs_idp,dd_idp, dd, idp_id_map, snp_id_map)

    # Full test set metric computation
    total_loss /= (batch_idx+1)
    acc /= (batch_idx+1)

    return total_loss.item(), acc

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