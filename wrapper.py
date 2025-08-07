import os, argparse, time, json, pickle
from src.train_eval import train_eval
from src.utils import plot_training_curves, plot_roc_curve, plot_precision_recall_curve, select_gpu, plot_uniqueness
from tabulate import tabulate
import numpy as np

# Parse args
def msg(name=None):
    return ''' COMICAL example runs:
        Basic run to train COMICAL
         >> python wrapper.py -fo <experimental run name> -gpu <gpu to run on> -out_flag <pairs> -dwn <0>
        Example run for clasf prediction using frozen encoders
        >> python wrapper.py -fo comical_PD -gpu 7 -target PD -out_flag clf -dwn 1 -lr 0.001
        Example run with hyperparameter tuning
        python wrapper.py -fo comical_top_1_tuned -tune 1 -top_n_perc 1 -ngpu 4
        '''

def parse_arguments():
    parser = argparse.ArgumentParser(usage=msg())
    # Data paths
    parser.add_argument("-ps", "--path_seq", dest='path_seq', action='store', help="Enter path for SNP sequences file", metavar="PS", default=os.path.join(os.getcwd(),'data','snp-encodings-from-vcf.csv'))
    parser.add_argument("-pi", "--path_idp", dest='path_idp', action='store', help="Enter path for IDPs file", metavar="PI", default=os.path.join(os.getcwd(),'data','T1_struct_brainMRI_IDPs.csv'))
    parser.add_argument("-pm", "--path_mod_b_map", dest='path_mod_b_map', action='store', help="Enter path for Modality B mapping file (optional)", metavar="PM", default=os.path.join(os.getcwd(),'data','T1mri.csv'))
    parser.add_argument("-pt", "--path_target_labels", dest='path_target_labels', action='store', help="Enter path for subject labels file", metavar="PT", default=os.path.join(os.getcwd(),'data','neuroDx_wPRS_real.csv')) #neuroDx_wPRS.csv
    parser.add_argument("-cov", "--path_covariates", dest='path_covariates', action='store', help="Enter path for covariates file", metavar="COV", default=os.path.join(os.getcwd(),'data','neuroDx_geneticPCs.csv'))
    parser.add_argument("-pmac", "--path_mod_a2group_map", dest='path_mod_a2group_map', action='store', help="Enter path for Modality A to latent grouping mapping file", metavar="PMAC", default=os.path.join(os.getcwd(),'data','SNPs_and_disease_mapping_with_pvalues.csv'))
    parser.add_argument("-pmbc", "--path_mod_b2group_map", dest='path_mod_b2group_map', action='store', help="Enter path for Modality B to latent grouping mapping file", metavar="PMBC", default=os.path.join(os.getcwd(),'data','IDPs_and_disease_mapping.csv'))
    parser.add_argument("-psp", "--path_saved_pairs", dest='path_saved_pairs', action='store', help="Enter path for previously saved pairings, or to save if no pairings exist", metavar="SAVEDPAIRS") #pairs_top_n_0.5
    parser.add_argument("-d", "--path_data", dest='path_data', action='store', help="Enter path for data directory, default is ./data", metavar="D", default=os.path.join(os.getcwd(),'data'))
    parser.add_argument("-p_sx", "--path_sub_idx", dest='path_sub_idx', action='store', help="Enter the path for the subject labels file.", metavar='SUBIDX', default=os.path.join(os.getcwd(),'data','clf_idxs.csv'))
    parser.add_argument("-p_sxu", "--path_sub_idx_unseen", dest='path_sub_idx_unseen', action='store', help="Enter the path for the unseen subject labels file.", metavar='SUBIDXU', default=os.path.join(os.getcwd(),'data','unseen_hc_idxs.csv'))

    # Results path
    parser.add_argument("-pr", "--path_res", dest='path_res', action='store', help="Output folder", metavar="PR", default=os.path.join(os.getcwd(),'results')) ##
    parser.add_argument("-pck", "--path_ckpt", dest='path_ckpt', action='store', help="Model Checkpoint folder", metavar="PCK", default='/fast/machad/comical/ray_results/train_2024-07-25_13-57-49/train_fc3fb_00000_0_lr=0.0019,warmup_steps=3800.2905,weight_decay=0.0162_2024-07-25_13-58-35/checkpoint_000009/checkpoint.pt |') 

    # Filenaming format
    parser.add_argument("-fo", "--fname_out_root", dest='fname_out_root', action='store', help='Enter prefix name for output files', metavar='FNAMEROOT', default='top5_emb_plotting') #Unamed_run

    # Training parameters
    parser.add_argument("-rnd_sed", "--random_seed", dest='random_seed', action='store', help='Enter random seed to be used for data splits', metavar='RNDSEED', default='42')
    parser.add_argument("-vsz", "--val_sz", dest='val_size', action='store', help='Enter percentage of data used for validation data split (eg. 20)', metavar='VALSZ', default='20')
    parser.add_argument("-tsz", "--test_sz", dest='test_size', action='store', help='Enter percentage of data used for test data split (eg. 10)', default='5')
    parser.add_argument("-gpu", "--gpu_nums", dest='gpu_nums', action='store', help="Enter the gpus to use", metavar="GPU")
    parser.add_argument("-ngpu", "--num_gpus_avail", dest='num_gpus_avail', action='store', help="Enter the number of gpus available", metavar="NGPU", default='1')
    parser.add_argument("-tune", "--tune_flag", dest='tune_flag', action='store', help="Enter 1 if you want to tune, 0 to just run experiments", metavar="TUNE", default='0')
    parser.add_argument("-gpu_tr", "--gpus_per_trial", dest='gpus_per_trial', action='store', help="Enter the number of gpus per trial to use", metavar="GPUTRIAL", default='1')
    parser.add_argument("-bz", "--batch_size", dest='batch_size', action='store', help="Enter the batch size", metavar="BZ", default='16384') #32768 #64 -> for clf, for pairs it throws nans
    parser.add_argument("-lr", "--learning_rate", dest='learning_rate', action='store', help="Enter the learning rate", metavar="LR", default='0.01') #0.05 - 0.001
    parser.add_argument("-e", "--epochs", dest='epochs', action='store', help="Enter the max epochs", metavar="EPOCHS", default='25') # 10 - 25
    parser.add_argument("-nl", "--num_layers", dest='num_layers', action='store', help="Enter the number of transformer layers", metavar="NUMLAY", default='2')
    parser.add_argument("-dm", "--d_model", dest='d_model', action='store', help="Enter the model dimensions", metavar="DIMS", default='64')
    parser.add_argument("-nh", "--n head", dest='nhead', action='store', help="Enter the number of heads on MHA", metavar="MHA", default='4')
    parser.add_argument("-df", "--dim_feedforward", dest='dim_feedforward', action='store', help="Enter the dimensions of the feedforward layer", metavar="DIMFF", default='64')#32
    parser.add_argument("-dp", "--dropout", dest='dropout', action='store', help="Enter the drop out decimal point", metavar="BZ", default='0.0')
    parser.add_argument("-u", "--units", dest='units', action='store', help="Enter the number of units in MLP hidden layer", metavar="BZ", default='16')
    parser.add_argument("-wd", "--weight_decay", dest='weight_decay', action='store', help="Enter the weight decay", metavar="WD", default='0.2')
    parser.add_argument("-al", "--alpha", dest='alpha', action='store', help="Enter the alpha value for the loss function", metavar="ALPHA", default='0.5')
    parser.add_argument("-be", "--beta", dest='beta', action='store', help="Enter the beta value for the loss function", metavar="BETA", default='0.5')
    
    # Run specifications
    parser.add_argument("-svemb", "--save_embeddings", dest='save_embeddings', action='store', help='Enter 1 if want to save embeddings', metavar='SVEMB', default='1') # 0
    parser.add_argument("-pltemb", "--plot_embeddings", dest='plot_embeddings', action='store', help='Enter 1 if want to plot embeddings, note this process can take a long time and a lot of memory.', metavar='PLTEMB', default='1')
    parser.add_argument("-top_n_perc", "--top_n_perc", dest='top_n_perc', action='store', help='Enter top n percentage of snps to use (e.g. 10%). Note: if not generating pairs, it must match the dataset top n value.', metavar='TOPN', default='5') # 0.5
    parser.add_argument("-resume", "--resume_from_batch", dest='resume_from_batch', action='store', help='Enter 1 if want to resume training from last batch checkpoint. Note: default = 0', metavar='RESUME', default='0')
    parser.add_argument("-ckpt_name", "--ckpt_name", dest='ckpt_name', action='store', help='Enter checkpoint name from batch to resume training.', metavar='ckpt_name', default='None')
    parser.add_argument("-dwn", "--downstream_pred_task_flag", dest='downstream_pred_task_flag', action='store', help='Enter 1 if want to train and evaluate with target based prediction (frozen encoders), used for downstream prediction.', metavar='SBP', default='0')
    parser.add_argument("-out_flag", "--out_flag", dest='out_flag', action='store', help='Enter clf for classification, reg for regression, or pairs for training encoders for pair association.', metavar='OUTFLAG', default='pairs') #pairs
    parser.add_argument("-target", "--target", dest='target', action='store', help='Enter the target to train classifier head.', metavar='TARGET', default='PD') # FAKE_PRS - neurological , PD_PRS
    parser.add_argument("-idx_col", "--index_col", dest='index_col', action='store', help='Enter the index column for the modality a and b data files.', metavar='IDXCOL', default='eid')
    parser.add_argument("-feat_a_idx_col", "--feat_a_index_col", dest='feat_a_index_col', action='store', help='Enter the index column for the modality a data file.', metavar='FEATAIDXCOL', default='SNPs')
    parser.add_argument("-feat_b_idx_col", "--feat_b_index_col", dest='feat_b_index_col', action='store', help='Enter the index column for the modality b data file.', metavar='FEATBIDXCOL', default='IDPs')
    parser.add_argument("-feat_a_target_col", "--feat_a_target_col", dest='feat_a_target_col', action='store', help='Enter the target column for the modality a data file.', metavar='FEATATARGETCOL', default='Disease')
    parser.add_argument("-feat_b_target_col", "--feat_b_target_col", dest='feat_b_target_col', action='store', help='Enter the target column for the modality b data file.', metavar='FEATBTARGETCOL', default='Disease')
    parser.add_argument("-cov_names", "--coveriate_names", dest='coveriate_names', action='store', help='Enter the names of the covariates to use. (no spaces between covariates and comma separated)', metavar='COVNAME', default='Age,Sex')
    parser.add_argument("-cbp", "--count_bins", dest='count_bins', action='store', help='Enter the number of bins to use for binning in the PLE tokenization.', metavar='CBP', default='64')
    parser.add_argument("-dec", "--decile", dest='decile', action='store', help='Enter the number of deciles to use for decile accuracy calculation, where 9 is the most strict, 1 is the least. Enter 0 for no decile accuracy calculation', metavar='DEC', default='0') #0.5
    parser.add_argument("-esf", "--early_stopper_flag", dest='early_stopper_flag', action='store', help='Enter 1 if want to use early stopping, 0 if not.', metavar='ESF', default='1')
    parser.add_argument("-sub_idx_flag", "--sub_idx_flag", dest='sub_idx_flag', action='store', help='Enter 1 if want to use a subset of the data for pair creation.', metavar='SUBIDXFLAG', default='0')
    parser.add_argument("-grad_clip", "--grad_clip", dest='grad_clip', action='store', help='Enter the gradient clipping value.', metavar='GRADCLIP', default='2.0')

    args = parser.parse_args()

    return args 

# Main
if __name__ == '__main__':
    begin_time = time.time()
    args = parse_arguments()

    ### Set discoverable GPU cards
    if args.gpu_nums is None:
        print('No GPU number given, selecting GPU with least memory usage.')
        args.gpu_nums = ','.join(select_gpu(int(args.num_gpus_avail),verbose=True))
    # For GPU - Pytorch
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_nums #model will be trained on GPU X
    print('Using GPU: ',args.gpu_nums)
    print('Current working directory: ',os.getcwd())
    ### Prepare environment ###
    # Create results directory if not exist
    if not os.path.isdir(args.path_res):
        os.mkdir(args.path_res)
        print('No result directory detected, results and checkpoints will be stored in: ',args.path_res)
    if not os.path.isdir(os.path.join(args.path_res,args.fname_out_root)):
        os.mkdir(os.path.join(args.path_res,args.fname_out_root))
        print(f'No results directory detected for {args.fname_out_root}, results and checkpoints will be stored in: {os.path.join(args.path_res,args.fname_out_root)}')
    # Create data directory if not exist
    if not os.path.isdir(os.path.join(os.getcwd(),'data')) and args.path_data != os.path.join(os.getcwd(),'data'):
        os.mkdir(os.path.join(os.getcwd(),'data'))
        print('No data directory detected, data and newly created pairs will be stored in: ',os.path.join(os.getcwd(),'data'))
    # Check if pairs exist
    if os.path.isfile(os.path.join(args.path_data,'pairs_top_n_'+args.top_n_perc+'.pickle')) or args.path_saved_pairs != None:
        pairs_exist = True
        print(f'Previously made pairs detected, pairs making will be skipped and pairs will be loaded.')
    else:
        pairs_exist = False
        print(f'No previously made pairs detected for {args.fname_out_root}, pairs will be generated and saved in this run.')

    # Check if checkpoint path was given, and if not, create one
    if args.path_ckpt == None:
        if not os.path.isdir(os.path.join(args.path_res,args.fname_out_root,'checkpoints')):
            os.mkdir(os.path.join(args.path_res,args.fname_out_root,'checkpoints'))
        path_ckpt = os.path.join(args.path_res,args.fname_out_root,'checkpoints')
        print(f'No checkpoint path was given, checkpoints for {args.fname_out_root} will be stored in: {os.path.join(os.getcwd(),"results",args.fname_out_root,"checkpoints")}')
    else:
        path_ckpt = args.path_ckpt

    ### Set dictionaries with paths and arguments to pass through framework ###
    paths = {
        'path_mod_a' : args.path_seq,
        'path_mod_b' : args.path_idp,
        'path_mod_b_map' : args.path_mod_b_map,
        'path_res' : args.path_res,
        'checkpoint_name' : path_ckpt,
        'tensorboard_log': os.path.join(args.path_res,args.fname_out_root,'tensorboard_logs'),
        'wd' : os.getcwd(),
        'path_target_labels' : args.path_target_labels,
        'path_covariates' : args.path_covariates,
        'path_mod_a2group_map' : args.path_mod_a2group_map,
        'path_mod_b2group_map' : args.path_mod_b2group_map,
        'path_saved_pairs' : args.path_saved_pairs if args.path_saved_pairs != None else os.path.join(args.path_data,'pairs_top_n_'+args.top_n_perc+'.pickle'),
        'path_data' : args.path_data,
        'path_sub_idx' : args.path_sub_idx,
        'path_sub_idx_unseen' : args.path_sub_idx_unseen if args.path_sub_idx_unseen != None else None,
    }

    run_args = {
        'alpha': float(args.alpha),
        'batch_size': int(args.batch_size),
        'beta': float(args.beta),
        'ckpt_name': args.ckpt_name,
        'count_bins': int(args.count_bins),
        'covariates_names': args.coveriate_names.split(','),
        'dim_feedforward': int(args.dim_feedforward),
        'target': args.target,
        'd_model': int(args.d_model),
        'decile': float(args.decile),
        'dropout': float(args.dropout),
        'early_stopper_flag':bool(int(args.early_stopper_flag)),
        'epochs': int(args.epochs),
        'feat_a_index_col':args.feat_a_index_col,
        'feat_b_index_col':args.feat_b_index_col,
        'feat_a_target_col':args.feat_a_target_col,
        'feat_b_target_col':args.feat_b_target_col,
        'fname_root_out': args.fname_out_root,
        'gpus_per_trial': args.gpus_per_trial,
        'grad_clip': float(args.grad_clip),
        'index_col': args.index_col if type(args.index_col) == str else int(args.index_col),
        'learning_rate': float(args.learning_rate),
        'nhead': int(args.nhead),
        'num_layers': int(args.num_layers),
        'out_flag': args.out_flag,
        'pairs_exist': pairs_exist,
        'plot_embeddings': bool(int(args.plot_embeddings)),
        'rnd_st': int(args.random_seed),
        'resume_from_batch': bool(int(args.resume_from_batch)),
        'save_embeddings': bool(int(args.save_embeddings)),
        'sub_idx_flag': bool(int(args.sub_idx_flag)),
        'downstream_pred_task_flag': bool(int(args.downstream_pred_task_flag)),
        'test_size': float(int(args.test_size) / 100),
        'top_n_perc': float(args.top_n_perc),
        'tune_flag': bool(int(args.tune_flag)),
        'units': int(args.units),
        'val_size': float(int(args.val_size) / 100),
        'weight_decay': float(args.weight_decay),
    }


    # Run training, hyperparam search and testing 
    results_dict = train_eval(paths,run_args)
    
    print(f'Test set loss {results_dict["metrics"]["loss_test"]}')
    # print(f'Test set top-1 accuracy {results_dict["metrics"]["acc_test"]}')
    print(f'Test set top-1 accuracy {results_dict["metrics"]["acc_test"]}') if args.out_flag == 'clf' else print(f'Test set R2 {results_dict["metrics"]["acc_test"]}')

    # Plot losses and result curves
    if bool(int(args.tune_flag)) == False:
        plot_training_curves(results_dict['data']['train_losses'], results_dict['data']['val_losses'],  results_dict['data']['val_accs'], os.path.join(args.path_res,args.fname_out_root))
        if args.out_flag == 'pairs':
            plot_uniqueness(results_dict['data']['uniqueness_a'], os.path.join(args.path_res,args.fname_out_root,'uniqueness_a.pdf'))
            plot_uniqueness(results_dict['data']['uniqueness_b'], os.path.join(args.path_res,args.fname_out_root,'uniqueness_b.pdf'))
        # FIXME: plotting currently externally to overlay all targets
        # if args.out_flag == 'clf':
        #     plot_roc_curve(np.asarray(results_dict['data']['test_preds']).astype('float'), np.asarray(results_dict['data']['test_labels']).astype('float').astype('int'), os.path.join(args.path_res,args.fname_out_root,'roc_curve.pdf'))
        #     plot_precision_recall_curve(np.asarray(results_dict['data']['test_preds']).astype('float'), np.asarray(results_dict['data']['test_labels']).astype('float').astype('int'), os.path.join(args.path_res,args.fname_out_root,'precision_recall_curve.pdf'))

    # Print hyperparameter configuration and results metrics
    print("Hyperparameter configuration and results metrics:")
    table_data = []
    # Add hyperparameter configuration to table data
    for key, value in results_dict['hyperparams'].items():
        table_data.append([key, value])

    # Add results metrics to table data
    for key, value in results_dict['metrics'].items():
        table_data.append([key, value])

    table = tabulate(table_data, headers=["Parameter", "Value"], tablefmt="grid")

    # Save table to txt file and print out
    with open(os.path.join(args.path_res,args.fname_out_root,'results_and_config_out.txt'), "w") as outfile:
        outfile.write(table)
    print(table)

   # Save results (data - deciles) as pickle file
    with open(os.path.join(args.path_res,args.fname_out_root,'results_data_and_config_out.pickle'), 'wb') as outfile:
        pickle.dump(results_dict, outfile)

    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))