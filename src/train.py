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
# from src.emb_plot import emb_plot
# TODO implement automatic hyperparameter tuning
import ray
from ray import tune
from ray.air import session
# from ray.air.checkpoint import Checkpoint
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import tempfile

from torch.utils.tensorboard import SummaryWriter
from src.models import comical, comical_new_emb, comical_new_emb_clasf,mlp_only
from src.utils import EarlyStopper, calculate_acc, calculate_decile_acc, emb_wg_plot, emb_wg_itm_plot
from sklearn.utils.class_weight import compute_class_weight
from tabulate import tabulate
from src.batch_sampler import UniquePairBatchSampler




def train(config, data=None, checkpoint_dir=None):
    # Tensorboard set  up
    writer = SummaryWriter(config['tensorboard_log_path'])
    tune = config['tune']
    # tune = True

    # Call functions from dataset to get pair buckets for evaluation of model (accuracy calculation)
    if config['out_flag'] == 'pairs':
        dd_seq_b, dd_seq_a = data.get_pairs_for_test()
        seq_b_id_map, seq_a_id_map = data.get_token_maps()

    # Define dataloaders
    # train_batch_sampler = UniquePairBatchSampler(data, config['train_index'], int(config["batch_size"]))
    # val_batch_sampler = UniquePairBatchSampler(data, config['val_index'], int(config["batch_size"])) # Adding batch sampler to avoid repeated pairs in matrix
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['train_index']), batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data,config['val_index']), batch_size=int(config["batch_size"]), shuffle=False)
    # train_loader = torch.utils.data.DataLoader(data, batch_sampler=train_batch_sampler) # batch_sampler is mutually exclusive with shuffle, and batch_size
    # val_loader = torch.utils.data.DataLoader(data, batch_sampler=val_batch_sampler)
    
    # Model and Hyperparams
    model = comical_new_emb(config) if config['out_flag']=='pairs' else mlp_only(config) if config['out_flag']=='mlp' else comical_new_emb_clasf(config)

    # If pre-trained model is used, load the weights
    # if config['out_flag'] == 'clf' or config['out_flag'] == 'reg':
    #     config['pretrained_model_path'] = '/home/machad/fast/comical/ray_results/train_2024-06-05_10-25-07/train_89522_00000_0_batch_size=4096,d_model=64,dim_feedforward=64,lr=0.0033,warmup_steps=236.5525,weight_decay=0.0015_2024-06-05_10-26-02/checkpoint_000009/checkpoint.pt'
    #     model.load_state_dict(torch.load(config['pretrained_model_path']))
    
    # Freeze encoders for pre-trained model
    if config['out_flag'] == 'clf' or config['out_flag'] == 'reg':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.h1.parameters():
            param.requires_grad = True
        for param in model.proj_down.parameters():
            param.requires_grad = True
    
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
    # optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'],betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'],betas=(0.9,0.98),eps=1e-6,weight_decay=config['weight_decay'])
    # optimizer = optim.AdamW(model.parameters(), lr=config['lr'],betas=(0.9,0.98),eps=1e-6)

    
    # Added classification and regression losses
    # if config['out_flag'] == 'clf' or config['out_flag'] == 'mlp':
    if config['out_flag'] in ['clf', 'mlp']:
        # Add class weights for imbalanced data on classification task
        class_weights=torch.tensor(compute_class_weight('balanced', classes=np.unique(data.get_labels()), y=data.get_labels()),dtype=torch.float).to(device)
        loss_clf = nn.CrossEntropyLoss(weight=class_weights)

    elif config['out_flag'] == 'reg':
        loss_reg = nn.MSELoss()
    else:
        loss_seq_a = nn.CrossEntropyLoss()
        loss_seq_b = nn.CrossEntropyLoss()
        # Updated loss with L_concept + L_condition
        # loss_seq_a_concept = nn.CrossEntropyLoss()
        # loss_seq_b_concept = nn.CrossEntropyLoss()
        # loss_seq_a_condition = nn.CrossEntropyLoss()
        # loss_seq_b_condition = nn.CrossEntropyLoss()
        alpha = config['alpha']
        beta = config['beta']

    # Learning rate scheduler with warm-up
    def lambda_lr(step):
        warmup_steps = config['warmup_steps']
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    

    # if config['resume_from_batch']:
    #     model, optimizer, start_batch = load_ckp(os.path.join(checkpoint_dir,config['last_checkpoint_path']), model, optimizer)
    # else:
    #     start_batch = 0

    if config['early_stopper_flag']:
        early_stopper = EarlyStopper(patience=10, min_delta=0.00001)

    train_losses = []
    val_losses = []
    val_accs = []
    uniqueness = {'seq_a_uniques': [], 'seq_b_uniques': []}
    # embs={'seq_a_embs':[], 'seq_b_embs':[]} 
    # Initialize dictionaries to hold embeddings associated with each unique ID.
    embs_a = {}  # This will store {'id_a': [seq_a_emb1, seq_a_emb2, ...]}
    embs_b = {}  # This will store {'id_b': [seq_b_emb1, seq_b_emb2, ...]}

    if tune:
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                print('checkpoint_dir',ray.train.get_checkpoint())
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                start = checkpoint_dict["epoch"]
                model.load_state_dict(checkpoint_dict["model_state"])


    # Training loop
    print('Training loop started')
    for epoch in tqdm(range(config['epochs']),desc='Training epoch loop'): 
        model.train()
        sum_loss = 0.      

        for batch_idx, (input_a_id,seq_a,input_b_id,seq_b,target,covariates) in enumerate(tqdm(train_loader,desc='Training batch loop')):
            # if batch_idx < start_batch:
            #     continue
            optimizer.zero_grad()
            seq_a,input_a_id,seq_b,input_b_id = seq_a.to(device).long(),input_a_id.to(device).long(),seq_b.to(device).long(),input_b_id.to(device).long()
            target = None if config['out_flag'] == 'pairs' else target.to(device).float()
            covariates = None if config['out_flag'] == 'pairs' else covariates.to(device).float()
            # Different outputs depending on whether we're saving embeddings
            if config['save_embeddings']:
                # Define how many embeddings per ID we want to keep
                # max_embs_per_id = config.get('max_embs_per_id', 5)
                max_embs_per_id = 5

                # Forward pass returns embeddings
                logits_seq_a, logits_seq_b, gen_emb, seq_b_emb = model(
                    seq_a, input_a_id, seq_b, input_b_id, config['save_embeddings']
                )

                # Collect up to `max_embs_per_id` embeddings for each id_a
                for id_a, emb in zip(input_a_id.cpu().numpy(), gen_emb):
                    emb_list = embs_a.get(id_a, [])
                    if len(emb_list) < max_embs_per_id:
                        emb_list.append(emb)
                        embs_a[id_a] = emb_list

                # Collect up to `max_embs_per_id` embeddings for each id_b
                for id_b, emb in zip(input_b_id.cpu().numpy(), seq_b_emb):
                    emb_list = embs_b.get(id_b, [])
                    if len(emb_list) < max_embs_per_id:
                        emb_list.append(emb)
                        embs_b[id_b] = emb_list

            else:
                if config['out_flag'] == 'pairs':
                    logits_seq_a, logits_seq_b = model(
                        seq_a, input_a_id, seq_b, input_b_id, config['save_embeddings']
                    )
                else:
                    pred = model(
                        seq_a, input_a_id, seq_b, input_b_id, config['save_embeddings'], covariates
                    )
            if config['out_flag'] == 'pairs':
                ground_truth = torch.arange(len(seq_a),dtype=torch.long,device=device)
                # ground_truth_conditioned = torch.arange(len(seq_a),dtype=torch.long,device=device)
                total_loss = (loss_seq_a(logits_seq_a,ground_truth) + loss_seq_b(logits_seq_b,ground_truth))/2
                # total_loss = alpha * (loss_seq_a_concept(logits_seq_a,ground_truth)+loss_seq_b_concept(logits_seq_b,ground_truth))/2 + beta * (loss_seq_a_condition(logits_seq_a,ground_truth_conditioned)+loss_seq_b_condition(logits_seq_b,ground_truth_conditioned))/2
            else:
                total_loss = loss_clf(pred,target.long()) if config['out_flag'] == 'clf' else loss_reg(pred.squeeze(-1),target)
            
            sum_loss += total_loss.item()
            total_loss.backward()
            if config['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config['grad_clip'], norm_type=2)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 2.0, norm_type=2) # gradient clipping to avoid vanishing/exploding gradients
            optimizer.step()
            scheduler.step()
            # save_ckp_batch(epoch,batch_idx,model,optimizer,checkpoint_dir)

        # loss per epoch
        sum_loss /= (batch_idx)
        
        if tune == False:
            train_losses.append(sum_loss)
            writer.add_scalar("Loss/train", sum_loss, epoch)

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        val_auc = 0.
        val_acc = 0.
        
        # Bring model to evaluation mode
        model.eval()
        with torch.no_grad():
            for batch_idx, (input_a_id,seq_a,input_b_id,seq_b,target,covariates) in enumerate(tqdm(val_loader,desc='Validation batch loop'), 0):
                    
                seq_a,input_a_id,seq_b,input_b_id = seq_a.to(device).long(),input_a_id.to(device).long(),seq_b.to(device).long(),input_b_id.to(device).long()
                target = None if config['out_flag'] == 'pairs' else target.to(device).float()
                covariates = None if config['out_flag'] == 'pairs' else covariates.to(device).float()
                if config['save_embeddings']:
                    logits_seq_a, logits_seq_b, gen_emb, seq_b_emb  = model(seq_a,input_a_id,seq_b,input_b_id,config['save_embeddings'])
                    # embs['seq_a_embs'].extend(gen_emb)
                    # embs['seq_b_embs'].extend(seq_b_emb)
                else:
                    if config['out_flag'] == 'pairs':
                        logits_seq_a, logits_seq_b  = model(seq_a,input_a_id,seq_b,input_b_id,config['save_embeddings'])
                    else:
                        pred  = model(seq_a,input_a_id,seq_b,input_b_id,config['save_embeddings'],covariates)
                if config['out_flag'] == 'pairs':
                    ground_truth = torch.arange(len(seq_a),dtype=torch.long,device=device)
                    total_val_loss = (loss_seq_a(logits_seq_a,ground_truth) + loss_seq_b(logits_seq_b,ground_truth))/2
                    # total_val_loss = alpha * (loss_seq_a_concept(logits_seq_a,ground_truth) + loss_seq_b_concept(logits_seq_b,ground_truth))/2 + beta * (loss_seq_a_condition(logits_seq_a,ground_truth) + loss_seq_b_condition(logits_seq_b,ground_truth))/2
                    # Compute softmax probs to use for accuracy calculation
                    probs_seq_a = np.argmax(logits_seq_a.softmax(dim=-1).cpu().numpy(), axis = 1)
                    probs_seq_b = np.argmax(logits_seq_b.softmax(dim=-1).cpu().numpy(), axis = 1)
                    # New accuracy calculation - overlap between top 10% of predicted values and true values rather than just the top predicted value
                    # probs_seq_a = logits_seq_a.softmax(dim=-1).cpu().numpy()
                    # probs_seq_b = logits_seq_b.softmax(dim=-1).cpu().numpy()
                    # Calculte accuracy per batch
                    val_acc += calculate_acc(seq_a.cpu().numpy(),input_a_id.cpu().numpy(),seq_b.cpu().numpy(),input_b_id.cpu().numpy(), probs_seq_a, probs_seq_b,dd_seq_b, dd_seq_a, seq_b_id_map, seq_a_id_map)
                    # val_acc += calculate_decile_acc(seq_a.cpu().numpy(),input_a_id.cpu().numpy(),seq_b.cpu().numpy(),input_b_id.cpu().numpy(), probs_seq_a, probs_seq_b,dd_seq_b, dd_seq_a, seq_b_id_map, seq_a_id_map, config['decile'])
                    # Calculate distribution of pairs for debugging every 10% of the validation set
                    if batch_idx % int(len(val_loader)/10) == 0:
                        seq_a_uniques, seq_a_counts = np.unique(probs_seq_a, return_counts=True)
                        seq_b_uniques, seq_b_counts = np.unique(probs_seq_b, return_counts=True)
                        # print(tabulate(zip(seq_a_uniques,seq_a_counts), headers=['seq_a_uniques','seq_a_counts']))
                        # print(tabulate(zip(seq_b_uniques,seq_b_counts), headers=['seq_b_uniques','seq_b_counts']))
                        uniqueness['seq_a_uniques'].extend(seq_a_uniques)
                        uniqueness['seq_b_uniques'].extend(seq_b_uniques)
                        
                else:
                    total_val_loss = loss_clf(pred,target.long()) if config['out_flag'] == 'clf' else loss_reg(pred.squeeze(-1),target)
                val_loss += total_val_loss
                val_steps += 1
                val_acc /= (batch_idx+1)
            print("Number of batches in validation loader",int(len(val_loader)))
            # Use if raytune is implemented
            if tune:
                # os.makedirs(os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}'), exist_ok=True)
                # torch.save(
                #     (model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}.pt'))
                # checkpoint = Checkpoint.from_directory(os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}.pt'))
                # session.report({"loss": (total_val_loss / val_steps)}, checkpoint=checkpoint)

                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(
                        {"epoch": epoch, "model_state": model.state_dict()},
                        os.path.join(tempdir, "checkpoint.pt"),
                    )
                    print("Saving checkpoint to", tempdir)
                    ray.train.report(metrics={"loss": (total_val_loss.cpu().numpy() / val_steps)}, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Save model and info per epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss,
                    },  os.path.join(checkpoint_dir,f'checkpoint_epoch_{epoch}.pt'))
                val_losses.append(total_val_loss.item()/val_steps)
                val_accs.append(val_acc)
                writer.add_scalar("Loss/val", total_val_loss.item()/val_steps, epoch)
            print(f'Training loss= {sum_loss} at epoch {epoch}')
            print(f'Vaidation loss= {total_val_loss.item()/val_steps} at epoch {epoch}')

        # Plot embedding weigths to show training progress - get the weights the embedding layers and send out to plot
        if config['plot_embeddings']:
            os.makedirs(os.path.join(os.getcwd(),'results',config['fname_root_out'],'emb_plots'), exist_ok=True)
            # Get the weights of the embedding layers
            # gen_emb_weight, snp_id_emb_weight, idp_emb_weight, idp_id_emb_weight = model.get_embeddings()
            gen_emb_weight, snp_id_emb_weight, idp_emb_weight, idp_id_emb_weight = model.gen_emb.weight.cpu().detach().numpy(), model.snp_id_emb.weight.cpu().detach().numpy(), model.idp_emb.weight.cpu().detach().numpy(), model.idp_id_emb.weight.cpu().detach().numpy()
            feats_a_map, feats_b_map = data.get_disease_idp_snp_map()
            # Plot the weights
            emb_wg_plot(os.path.join(os.getcwd(),'results',config['fname_root_out'],'emb_plots'),[gen_emb_weight, snp_id_emb_weight, idp_emb_weight, idp_id_emb_weight], epoch, dd_seq_b, dd_seq_a, seq_b_id_map, seq_a_id_map, feats_a_map, feats_b_map)
            # Plot by embeddings
            emb_wg_itm_plot(os.path.join(os.getcwd(),'results',config['fname_root_out'],'emb_plots'), [embs_a,embs_b],epoch, dd_seq_b, dd_seq_a, seq_b_id_map, seq_a_id_map, feats_a_map, feats_b_map)


        # Early stopping
        if config['early_stopper_flag']:
            if early_stopper.early_stop(total_val_loss):
                print("Early stopping")
                break
    print("Finished Training")
    if tune == False:
        # Clear tensorboard variables
        writer.flush()
        writer.close()
        # Save embeddings
        if config['save_embeddings']:
            with open(os.path.join(os.getcwd(),'results',config['results_path'],'embs.pickle'), "wb") as outfile:
                pickle.dump([embs_a,embs_b],outfile)
        # if config['plot_embeddings']: # Deprecated - too resource intensive
        #     emb_plot(os.path.join(os.getcwd(),'results',config['fname_root_out']),embs)

        if config['out_flag'] == 'clf':
            uniqueness = None
        return train_losses, val_losses, val_accs, uniqueness