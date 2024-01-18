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
import pickle
from src.emb_plot import emb_plot
# TODO implement automatic hyperparameter tuning
# import ray
# from ray import tune
# from ray.air import session
# from ray.air.checkpoint import Checkpoint
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune import CLIReporter

from src.test import test_model

from src.utils  import idp_tokenization

from torch.utils.tensorboard import SummaryWriter
from src.models import comical, comical_new_emb



def train(config, data=None, checkpoint_dir=None):
    # Tensorboard set  up
    writer = SummaryWriter(config['tensorboard_log_path'])
    tune = config['tune']

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['train_index']), batch_size=int(config["batch_size"]), shuffle=True)

    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['val_index']), batch_size=int(config["batch_size"]), shuffle=True)
    
    # Model and Hyperparams
    model = comical_new_emb(config)
    
    # GPU settings
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            # model = DDP(model, device_ids=[rank]) # TODO: implement DDP for better paralelization (if needed)
    model = model.to(device)
    
    ## TODO: Priority: implement automatic checkpointing system, where model training can pick up from previous checkpoint.
    ## Probably will need to checkpoint for epochs batches rather than epochs
    ## Implement try and catch for assertion on model

    # def save_ckp_batch(epoch, batch_idx, model, optimizer, checkpoint_dir):
    #     torch.save({
    #         'batch': batch_idx,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #     },  os.path.join(checkpoint_dir,f'checkpoint_epoch{epoch}_batch{batch_idx}'))

    # def load_ckp(checkpoint_fpath, model, optimizer):
    #     checkpoint = torch.load(checkpoint_fpath)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     return model, optimizer, int(checkpoint['batch'])

    # Optimizer using clip settings (could be updated / added to hyperparmeter configuration)
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

    loss_seq = nn.CrossEntropyLoss()
    loss_idp = nn.CrossEntropyLoss()

    # if config['resume_from_batch']:
    #     model, optimizer, start_batch = load_ckp(os.path.join(checkpoint_dir,config['last_checkpoint_path']), model, optimizer)
    # else:
    #     start_batch = 0

    train_losses = []
    val_losses = []
    embs={'gen_embs':[], 'idp_embs':[]}

    # Training loop
    print('Training loop started')
    for epoch in tqdm(range(config['epochs']),desc='Training epoch loop'): 
        model.train()
        sum_loss = 0.      

        for batch_idx, (snp_id,seq,idp_id,idp) in enumerate(tqdm(train_loader,desc='Training batch loop')):
            # if batch_idx < start_batch:
            #     continue
            optimizer.zero_grad()
            seq,snp_id,idp,idp_id = seq.to(device).long(),snp_id.to(device).long(),idp.to(device).long(),idp_id.to(device).long()
            # Different outputs depening if returning embeddings
            if config['save_embeddings']:
                logits_seq, logits_idp, gen_emb, idp_emb  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
                embs['gen_embs'].extend(gen_emb)
                embs['idp_embs'].extend(idp_emb)
            else:
                logits_seq, logits_idp  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])

            ground_truth = torch.arange(len(seq),dtype=torch.long,device=device)
            total_loss = (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
            sum_loss += total_loss.item()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2.0, norm_type=2) # gradient clipping to avoid vanishing/exploding gradients
            optimizer.step()
            save_ckp_batch(epoch,batch_idx,model,optimizer,checkpoint_dir)

        # loss per epoch
        sum_loss /= (batch_idx)
        
        if tune == False:
            train_losses.append(sum_loss)
            writer.add_scalar("Loss/train", sum_loss, epoch)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_auc = 0.
        # Bring model to evaluation mode
        model.eval()
        with torch.no_grad():
            for batch_idx, (snp_id,seq,idp_id,idp) in enumerate(tqdm(val_loader,desc='Validation batch loop'), 0):
                    
                seq,snp_id,idp,idp_id = seq.to(device).long(),snp_id.to(device).long(),idp.to(device).long(),idp_id.to(device).long()
                if config['save_embeddings']:
                    logits_seq, logits_idp, gen_emb, idp_emb  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])
                    embs['gen_embs'].extend(gen_emb)
                    embs['idp_embs'].extend(idp_emb)
                else:
                    logits_seq, logits_idp  = model(seq,snp_id,idp,idp_id,config['save_embeddings'])

                ground_truth = torch.arange(len(seq),dtype=torch.long,device=device)
                total_val_loss = (loss_seq(logits_seq,ground_truth) + loss_idp(logits_idp,ground_truth))/2
                val_loss += total_val_loss
                val_steps += 1
        
            # Use if raytune is implemented
            if tune:
                os.makedirs("my_model", exist_ok=True)
                torch.save(
                    (model.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
                checkpoint = Checkpoint.from_directory("my_model")
                session.report({"loss": (total_val_loss / val_steps)}, checkpoint=checkpoint)
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Save model and info per epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    },  os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}'))
                val_losses.append(total_val_loss.item()/val_steps)
                writer.add_scalar("Loss/val", total_val_loss.item()/val_steps, epoch)

    print("Finished Training")
    if tune == False:
        # Clear tensorboard variables
        writer.flush()
        writer.close()
        # Save embeddings
        if config['save_embeddings']:
            with open(os.path.join(os.getcwd(),'results',config['results_path'],'embs.pickle'), "wb") as outfile:
                pickle.dump(embs,outfile)
        if config['plot_embeddings']:
            emb_plot(os.path.join(os.getcwd(),'results',config['fname_root_out']),embs)


        return train_losses, val_losses