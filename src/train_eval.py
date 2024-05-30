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
# import ray
# from ray import tune
# from ray.air import session, RunConfig
# from ray.air.checkpoint import Checkpoint
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune import CLIReporter


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
            'warmup_steps' : 2000, # using 2000 as recommended in clip paper
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
            "lr": tune.grid_search([0.001,0.0001,0.00001]),
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
            'warmup_steps' : 2000, # using 2000 as recommended in clip paper
        }
        scheduler = ASHAScheduler(
            max_t=10,
            grace_period=3,
            reduction_factor=3,
            metric='auc',
            mode='max',
        )

        reporter = CLIReporter(
            metric_columns=["loss", "auc", "training_iteration"])

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train, data = data),
                resources={"cpu": 24, "gpu": float(args.gpus_per_trial)}
            ),
            tune_config=tune.TuneConfig(
                metric="auc",
                mode="max",
                scheduler=scheduler,
                num_samples=1,
            ),
            param_space=config,
            run_config=RunConfig(local_dir="./ray_results")
        )
        # Run hyperparameter tuning and report training results
        ray.init(_temp_dir='/home/machad/fast/comical/ray_tmp')
        result = tuner.fit()
        best_trial = result.get_best_result("auc", "max")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final AUC: {}".format(best_trial.metrics["auc"]))
        print("Best trial final validation loss: {}".format(best_trial.metrics["loss"]))
        # Prepare to predict on test set using best trial
        config = best_trial.config
        best_checkpoint_path = os.path.join(os.path.dirname(best_trial.checkpoint._local_path),'my_model','checkpoint.pt')

    else:
        train_losses, val_losses, val_accs, uniqueness = train(config, data=data, checkpoint_dir = paths['checkpoint_name'])
        # Select checkpoint with the lowest loss on validation set
        best_epoch = np.argmin(val_losses)
        best_checkpoint_path = os.path.join(paths['checkpoint_name'], f'checkpoint_epoch_{best_epoch}')

    ## Evaluate model ##
    # Evaluate on all data paritions - using best checkpointed model
    # loss_train, acc_train, _ = test_model(config, data, train_idx, best_checkpoint_path)
    # loss_val, acc_val, _ = test_model(config, data, val_idx, best_checkpoint_path)
    loss_test, acc_test, extra_test = test_model(config, args ,data, test_idx, best_checkpoint_path)

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
            'train_losses': train_losses,
            'val_losses':val_losses,
            'val_accs':val_accs,
            'test_preds': [str(i) for i in extra_test['preds']],
            'test_labels': [str(i) for i in extra_test['target']],
            'uniqueness_a' : uniqueness['seq_a_uniques'],
            'uniqueness_b' : uniqueness['seq_b_uniques'],
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
