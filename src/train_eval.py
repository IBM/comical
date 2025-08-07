import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.train import train
from src.test import test_model
# from legacy.dataset import dataset
from src.dataset_template import dataset

# Imports for hyperparameter tuning
import ray
from ray import tune
from ray.air import session, RunConfig
# from ray.air.checkpoint import Checkpoint
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

import statsmodels.api as sm
import seaborn as sns


def train_eval(paths, args, config = None):
    # Load data and create dataset object
    data = dataset(paths, args)
    # Get data splits and confirm there is no overlap between them.
    train_idx, val_idx, test_idx = data.get_data_splits()
    
    print('# of training pairs:', len(train_idx))
    print('# of validation pairs:', len(val_idx))
    print('# of testing pairs:', len(test_idx))
    
    assert np.intersect1d(train_idx, val_idx).size == 0 and np.intersect1d(train_idx, test_idx).size == 0 and np.intersect1d(val_idx, test_idx).size == 0
   
    # Set hyperparameters
    if config == None:
        config = {
            # Data indices
            'train_index' : train_idx,
            'val_index' : val_idx,
            'test_index' : test_idx,
            'tune' : args['tune_flag'],
            ## Training hyperparms
            "lr": args['learning_rate'],
            'weight_decay': args['weight_decay'],
            "batch_size" : args['batch_size'],
            "epochs":args['epochs'],
            ## Model hyperparams
            'units': args['units'],
            'num_layers' : args['num_layers'],
            'd_model' : args['d_model'], 
            'nhead': args['nhead'],
            'dim_feedforward' : args['dim_feedforward'],
            'dropout' : args['dropout'],
            'layer_norm_eps' : 0.000001,
            'activation' : 'gelu',
            ## Model parameters (not trainable)
            'tensorboard_log_path':paths['tensorboard_log'],
            'fname_root_out':args['fname_root_out'],
            'num_snps':10000,
            'num_idps':139,
            'idp_tok_dims':64,
            # Model options
            'save_embeddings':args['save_embeddings'],
            'plot_embeddings':args['plot_embeddings'],
            'results_path':os.path.join(os.getcwd(),'results',args['fname_root_out']),
            'resume_from_batch':args['resume_from_batch'],
            'ckpt_name':args['ckpt_name'],
            'subject_based_pred_flag':args['downstream_pred_task_flag'],
            'out_flag':args['out_flag'],
            'target':args['target'],
            'warmup_steps' : 2000, # using 2000 as recommended in clip paper, 
            'num_classes': 1,#2 if args['out_flag'] == 'clf' else 1,
            'decile':args['decile'],
            'early_stopper_flag':args['early_stopper_flag'],
            'alpha':args['alpha'],
            'beta':args['beta'],
            'grad_clip':args['grad_clip'],
        }
    ### Train model ###
    if args['tune_flag']:
        print('Tuning hyperparameters')
        config = {
            # Data indices
            'train_index' : train_idx,
            'val_index' : val_idx,
            'test_index' : test_idx,
            'tune' : args['tune_flag'],
            ## Training hyperparms
            "lr": tune.loguniform(1e-5, 1e-2) ,#if args['out_flag'] == 'pairs' else 0.0001, #0.0001
            'weight_decay': tune.loguniform(1e-3, 1e-1),
            "batch_size" : tune.grid_search([4096,32768]) if args['out_flag'] == 'pairs' else 1024,
            "epochs":args['epochs'],
            ## Model hyperparams
            'units': args['units'],
            'num_layers' : args['num_layers'],
            'd_model' : tune.grid_search([64,128,256]) if args['out_flag'] == 'pairs' else 64, 
            'nhead': args['nhead'],
            'dim_feedforward' : tune.grid_search([64,128,256]) if args['out_flag'] == 'pairs' else 128,
            'dropout' : args['dropout'],
            'layer_norm_eps' : 0.000001,
            'activation' : 'gelu',
            ## Model parameters (not trainable)
            'tensorboard_log_path':paths['tensorboard_log'],
            'num_snps':10000,
            'num_idps':139,
            'idp_tok_dims':64,
            # Model options
            'save_embeddings':args['save_embeddings'],
            'plot_embeddings':args['plot_embeddings'],
            'results_path':os.path.join(os.getcwd(),'results',args['fname_root_out']),
            'resume_from_batch':args['resume_from_batch'],
            'ckpt_name':args['ckpt_name'],
            'subject_based_pred_flag':args['downstream_pred_task_flag'],
            'out_flag':args['out_flag'],
            'target':args['target'],
            'warmup_steps' : tune.loguniform(1e2, 1e4), # using 2000 as recommended in clip paper,
            'num_classes': 2 if args['out_flag'] == 'clf' else 1,
            'decile':args['decile'],
            'early_stopper_flag':args['early_stopper_flag'],
            'alpha':tune.uniform(0.1, 0.9),
            'beta':tune.uniform(0.1, 0.9),
        }
        scheduler = ASHAScheduler(
            max_t=10,
            grace_period=3,
            reduction_factor=3,
            metric='loss',
            mode='min',
        )

        reporter = CLIReporter(
            metric_columns=["loss", "auc", "training_iteration"])

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train, data = data),
                # resources={"cpu": 24, "gpu": float(args['gpus_per_trial'])}
                resources={"cpu": 16, "gpu": float(args['gpus_per_trial'])}
            ),
            tune_config=tune.TuneConfig(
                # metric="auc", # dealt with in scheduler
                # mode="max",
                scheduler=scheduler,
                num_samples=1,
            ),
            param_space=config,
            run_config=RunConfig(storage_path =os.path.abspath("./ray_results")) # used to be local_dir, but new version of ray uses storage_path
        )
        # Run hyperparameter tuning and report training results
        ray.init(_temp_dir='/home/machad/fast/comical/ray_tmp')
        result = tuner.fit()
        best_trial = result.get_best_result("loss", "min")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final AUC: {}".format(best_trial.metrics["loss"]))
        print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))
        # Prepare to predict on test set using best trial
        config = best_trial.config
        # best_checkpoint_path = os.path.join(os.path.dirname(best_trial.checkpoint._local_path),'my_model','checkpoint.pt')
        best_checkpoint_path = os.path.join(best_trial.checkpoint.path,'checkpoint.pt')

    else:
        train_losses, val_losses, val_accs, uniqueness = train(config, data=data, checkpoint_dir = paths['checkpoint_name'])
        # Select checkpoint with the lowest loss on validation set
        # best_epoch = np.argmin(val_losses)
        # best_epoch = np.argmin(val_losses)+int(len(val_losses)*0.2)# balancing convergence and overfitting
        # best_checkpoint_path = os.path.join(paths['checkpoint_name'], f'checkpoint_epoch_{best_epoch}.pt')

        # best_checkpoint_path = paths['checkpoint_name']
        best_checkpoint_path = '/home/machad/fast/comical/data/BD_10_checkpoint_epoch_2499.pt'
        # train_losses = [0.1,0.2,0.3,0.4,0.5]
        # val_losses = [0.1,0.2,0.3,0.4,0.5]
        # val_accs = [0.1,0.2,0.3,0.4,0.5]
        # uniqueness = None

    ## Evaluate model ##
    # Evaluate on all data paritions - using best checkpointed model
    # loss_train, acc_train, _ = test_model(config, data, train_idx, best_checkpoint_path)
    # loss_val, acc_val, _ = test_model(config, data, val_idx, best_checkpoint_path)
    loss_test, acc_test, extra_test, softmax_idp, softmax_snp = test_model(config, args ,data, test_idx, best_checkpoint_path)

    # Plot partregress and regplot if PRS is the target
    if args['out_flag'] == 'prs':
        pred_prs = sm.add_constant(extra_test['preds'])
        sm_model = sm.OLS(extra_test['target'],pred_prs)
        sm_results = sm_model.fit()
        # fig = sm.graphics.plot_partregress_grid(sm_results)
        fig = sm.graphics.plot_fit(sm_results, 'x1')
        fig.tight_layout(pad=1.0)
        fig.savefig(os.path.join(config['results_path'],'partregress.png'))

        
        fig = sns.regplot(x=extra_test['preds'], y=extra_test['target'])
        fig.savefig(os.path.join(config['results_path'],'regplot.png'))

    # Return dictionary with results, data info
    results_ret = {
        'metrics':{
            # 'loss_train':loss_train,
            # 'loss_val':loss_val,
            'loss_test':loss_test,
            # 'acc_train':acc_train,
            # 'acc_val':acc_val,
            'acc_test':acc_test,
        },
        'data':{
            'train_losses': train_losses if args['tune_flag'] == False else None,
            'val_losses':val_losses if args['tune_flag'] == False else None,
            'val_accs':val_accs if args['tune_flag'] == False else None,
            'test_preds': [str(i) for i in extra_test['preds']],
            'test_labels': [str(i) for i in extra_test['target']],
            'uniqueness_a' : uniqueness['seq_a_uniques'] if args['tune_flag'] == False and uniqueness is not None else None, 
            'uniqueness_b' : uniqueness['seq_b_uniques'] if args['tune_flag'] == False and uniqueness is not None else None,
            'softmax_idp' : softmax_idp,
            'softmax_snp' : softmax_snp,
        },
        'hyperparams':{
            'lr':config["lr"],
            'batch_size':config["batch_size"],
            'units':config["units"],
            'd_model':config["d_model"],
            'nhead':config["nhead"],
            'dim_feedforward':config["dim_feedforward"],
            'dropout':config["dropout"],
            'layer_norm_eps':config["layer_norm_eps"],
            'activation':config["activation"],
            'checkpoint_path':best_checkpoint_path
        }
    }
    return results_ret
