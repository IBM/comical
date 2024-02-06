import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.train import train
from src.test import test_model
from src.dataset import dataset
from datasets import load_dataset


def train_eval(paths, args, config = None):
    # Load data and create dataset
    # For experiments with COCO. TODO: Delete for final release.
    if args['train_coco'] == False:
        from src.train import train
        from src.test import test_model
        # Load and prepare data
        data = dataset(paths, args)
        # Get data splits and confirm there is no overlap between them.
        train_idx, val_idx, test_idx = data.get_data_splits()

        print('# of training pairs:', len(train_idx))
        print('# of validation pairs:', len(val_idx))
        print('# of testing pairs:', len(test_idx))

        assert np.intersect1d(train_idx, val_idx).size == 0 and np.intersect1d(train_idx, test_idx).size == 0 and np.intersect1d(val_idx, test_idx).size == 0
    else:
        from src.train_coco import train
        from src.test_coco import test_model

        # Evaluating framework with COCO
        data_args = {}
        model_args = {}
        training_args = {}
        data_args['dataset_name'] = 'ydshieh/coco_dataset_script'
        data_args['dataset_config_name'] = '2017'
        model_args['cache_dir'] = '/dccstor/ukb-pgx/comical/CLIP/.cache'
        data_args['data_dir'] = '/dccstor/ukb-pgx/comical/CLIP/data'
        model_args['use_auth_token'] = False
        model_args['model_name_or_path'] = './clip-roberta'
        model_args['use_fast_tokenizer'] = True
        model_args['freeze_vision_model'] = False
        model_args['freeze_text_model'] = False
        training_args['seed'] = 42

        data = load_dataset(
            data_args['dataset_name'],
            data_args['dataset_config_name'],
            cache_dir=model_args['cache_dir'],
            keep_in_memory=False,
            data_dir=data_args['data_dir'],
            use_auth_token=True if model_args['use_auth_token'] else None,
        )
        
        train_idx = data['train']['image_id']
        val_idx = data['validation']['image_id']
        test_idx = data['test']['image_id']

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
            'subject_based_pred_flag':args['subject_based_pred_flag'],
            'out_flag':args['out_flag'],
            'disease':args['disease'],
        }
    # Train model
    train_losses, val_losses = train(config, data=data, checkpoint_dir = paths['checkpoint_name'])
    # Select checkpoint with the lowest loss on validation set
    best_epoch = np.argmin(val_losses)
    best_checkpoint_path = os.path.join(paths['checkpoint_name'], f'checkpoint_epoch_{best_epoch}')


    # Evaluate on all data paritions - using best checkpointed model
    # loss_train, acc_train, _ = test_model(config, data, train_idx, best_checkpoint_path)
    # loss_val, acc_val, _ = test_model(config, data, val_idx, best_checkpoint_path)
    loss_test, acc_test, extra_test = test_model(config, data, test_idx, best_checkpoint_path)

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
        },
        'data_auc_plot':{
            'test_preds': [str(i) for i in extra_test['preds']],
            'test_labels': [str(i) for i in extra_test['target']],
        }
    }
    return results_ret
