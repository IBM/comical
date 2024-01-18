from functools import partial
from re import A
from tkinter.messagebox import NO
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.test import test_model

from pprint import pprint

# from utils  import compute_auc

from torch.utils.tensorboard import SummaryWriter
from src.models import comical

import clip
from PIL import Image



def train(config, data=None, checkpoint_dir=None):
    writer = SummaryWriter(config['tensorboard_log_path'])
    tune = config['tune']

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(data['train'], batch_size=int(config["batch_size"]), shuffle=True)

    val_loader = torch.utils.data.DataLoader(data['validation'], batch_size=int(config["batch_size"]), shuffle=True)
    
    
    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
            model = nn.DataParallel(model)
            # model = DDP(model, device_ids=[rank])
    # Model and Hyperparams
    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    
    def convert_models_to_fp32(model): 
        for p in model.parameters(): 
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float() 


    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    # optimizer = optim.Adam(model.parameters(), lr=config["lr"]) 
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    loss_seq = nn.CrossEntropyLoss()
    loss_idp = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []


    # Training loop
    print('Training loop started')
    for epoch in range(config['epochs']): 
        model.train()
        sum_loss = 0.      

        for batch_idx, data_dict in enumerate(tqdm(train_loader)):
            images = []
            # for path in tqdm(data_dict['image_path'],desc='image preprocess training set'):
            for path in data_dict['image_path']:
                images.append(preprocess(Image.open(os.path.join('/dccstor/ukb-pgx/comical/CLIP/',path))))
            text,images = clip.tokenize(data_dict['caption']).to(device), torch.stack(images).to(device)
            optimizer.zero_grad()   
            # images,text = images.to(device),text.to(device)
            logits_seq, logits_idp  = model(images,text)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_loss = (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
            sum_loss += total_loss.item()
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            if batch_idx== 10:
                break

        # loss and auc per epoch
        sum_loss /= (batch_idx)
        
        if tune == False:
            train_losses.append(sum_loss)
            writer.add_scalar("Loss/train", sum_loss, epoch)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_auc = 0.
        # bring models to evaluation mode
        model.eval()
        # with torch.no_grad():
        for batch_idx, data_dict in enumerate(val_loader, 0):
            images = []
            for path in data_dict['image_path']:
                images.append(preprocess(Image.open(os.path.join('/dccstor/ukb-pgx/comical/CLIP/',path))))
            text,images = clip.tokenize(data_dict['caption']).to(device), torch.stack(images).to(device)
                
            images, text = images.to(device), text.to(device)
            logits_seq, logits_idp = model(images, text)


            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            total_val_loss = (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
            val_loss += total_val_loss
            val_steps += 1
    
        if tune:
            os.makedirs("my_model", exist_ok=True)
            torch.save(
                (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
            checkpoint = Checkpoint.from_directory("my_model")
            session.report({"loss": (total_val_loss / val_steps)}, checkpoint=checkpoint)
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}'))
            val_losses.append(total_val_loss.item()/val_steps)
            writer.add_scalar("Loss/val", total_val_loss.item()/val_steps, epoch)

    print("Finished Training")
    if tune == False:
        writer.flush()
        writer.close()
        return train_losses, val_losses