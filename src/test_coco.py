import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils import compute_auc
from src.models import comical
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image



def test_model(config, data=None, subset_index=None, best_checkpoint_name=None):    
    # Define test loader
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data['test'],subset_index), batch_size=int(config['batch_size']), shuffle=True)

    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    # Define model   
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) 

    # Load state for best model
    if config['tune']:
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_name, ".pt"))

    else:
        model_state, optimizer_state = torch.load(best_checkpoint_name)

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    model_state['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    model_state['model_state_dict']["context_length"] = model.context_length # default is 77
    model_state['model_state_dict']["vocab_size"] = model.vocab_size 

    model.load_state_dict(model_state['model_state_dict'])


    # Run testing
    model.eval()
    total_loss = 0.
    acc = 0.
    seq_l = []
    idp_l = []
    loss_seq = nn.CrossEntropyLoss()
    loss_idp = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(tqdm(test_loader)):
            images = []
            for path in tqdm(data_dict['image_path'],desc='image preprocess'):
                images.append(preprocess(Image.open(os.path.join('/dccstor/ukb-pgx/comical/CLIP/',path))))
            text,images = clip.tokenize(data_dict['caption']).to(device), torch.stack(images).to(device)

            images,text = images.to(device),text.to(device)
            logits_seq, logits_idp  = model(images,text)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss += (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2

            probs_seq = np.argmax(logits_seq.softmax(dim=-1).cpu().numpy(), axis = 1)
            probs_idp = np.argmax(logits_idp.softmax(dim=-1).cpu().numpy(), axis = 1)

            # acc += (np.sum(probs_seq == range(5))/5 + np.sum(probs_idp == range(5))/5)/2
            acc += (np.sum(probs_seq == range(len(images)))/len(images) + np.sum(probs_idp == range(len(images)))/len(images))/2

    # Dataloader metric computation
        total_loss /= (batch_idx+1)
        acc /= (batch_idx+1)

    return total_loss.item(), acc