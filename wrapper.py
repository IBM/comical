import os, argparse, time, json
from src.train_eval import train_eval

# Parse args
def msg(name=None):
    return ''' COMICAL example runs:
        Basic run
         >> python wrapper.py -fo comical_train -gpu 0 -tune 0
        Example run for debugging session on RPI
         >> python -m debugpy --listen 0.0.0.0:1326 --wait-for-client wrapper.py -tr_coco 0 -e 1 -bz 7500 -fo top_01_perc_7500bz_1e - top_n_perc 0.1
        Example run from cli on CCC once in comptue node with environment activated
        python wrapper.py -fo comical_new_top1snps_pairs -bz 7500 -tr_coco 0 -e 4
        '''

def parse_arguments():
    parser = argparse.ArgumentParser(usage=msg())
    # Data paths
    parser.add_argument("-ps", "--path_seq", dest='path_seq', action='store', help="Enter path for SNP sequences file", metavar="PS", default=os.path.join(os.getcwd(),'data','snp-encodings-from-vcf.csv'))
    parser.add_argument("-pi", "--path_idp", dest='path_idp', action='store', help="Enter path for IDPs file", metavar="PI", default=os.path.join(os.getcwd(),'data','T1_struct_brainMRI_IDPs.csv'))
    parser.add_argument("-pm", "--path_id_map", dest='path_idp_map', action='store', help="Enter path for IDPs name mapping file", metavar="PM", default=os.path.join(os.getcwd(),'data','T1mri.csv'))
    parser.add_argument("-pp", "--path_pairs", dest='path_pairs', action='store', help="Enter path for pairing matches file", metavar="PP", default=os.path.join(os.getcwd(),'data','pairs.csv'))
    parser.add_argument("-pt", "--path_subj_labels", dest='path_subj_labels', action='store', help="Enter path for subject labels file", metavar="PT", default=os.path.join(os.getcwd(),'data','neuroDx.csv'))

    # Results path
    parser.add_argument("-pr", "--path_res", dest='path_res', action='store', help="Output folder", metavar="PR", default=os.path.join(os.getcwd(),'results')) 
    parser.add_argument("-pck", "--path_ckpt", dest='path_ckpt', action='store', help="Model Checkpoint folder", metavar="PCK") 

    # Filenaming format
    parser.add_argument("-fo", "--fname_out_root", dest='fname_out_root', action='store', help='Enter prefix name for output files', metavar='FNAMEROOT', default='Unamed_run')

    # Training parameters
    parser.add_argument("-rnd_sed", "--random_seed", dest='random_seed', action='store', help='Enter random seed to be used for data splits', metavar='RNDSEED', default='42')
    parser.add_argument("-vsz", "--val_sz", dest='val_size', action='store', help='Enter percentage of data used for validation data split (eg. 20)', metavar='VALSZ', default='20')
    parser.add_argument("-tsz", "--test_sz", dest='test_size', action='store', help='Enter percentage of data used for test data split (eg. 10)', default='10')
    parser.add_argument("-gpu", "--gpu_nums", dest='gpu_nums', action='store', help="Enter the gpus to use", metavar="GPU", default='2,3,4,5,6,7')
    parser.add_argument("-tune", "--tune_flag", dest='tune_flag', action='store', help="Enter 1 if you want to tune, 0 to just run experiments", metavar="TUNE", default='0')
    parser.add_argument("-gpu_tr", "--gpus_per_trial", dest='gpus_per_trial', action='store', help="Enter the number of gpus per trial to use", metavar="GPUTRIAL", default='1')
    parser.add_argument("-bz", "--batch_size", dest='batch_size', action='store', help="Enter the batch size", metavar="BZ", default='256')
    parser.add_argument("-lr", "--learning_rate", dest='learning_rate', action='store', help="Enter the learning rate", metavar="LR", default='0.00001')
    parser.add_argument("-e", "--epochs", dest='epochs', action='store', help="Enter the max epochs", metavar="EPOCHS", default='15')
    parser.add_argument("-nl", "--num_layers", dest='num_layers', action='store', help="Enter the number of transformer layers", metavar="NUMLAY", default='2')
    parser.add_argument("-dm", "--d_model", dest='d_model', action='store', help="Enter the model dimensions", metavar="DIMS", default='64')
    parser.add_argument("-nh", "--nhead", dest='nhead', action='store', help="Enter the number of heads on MHA", metavar="MHA", default='4')
    parser.add_argument("-df", "--dim_feedforward", dest='dim_feedforward', action='store', help="Enter the dimensions of the feedforward layer", metavar="DIMFF", default='32')
    parser.add_argument("-dp", "--dropout", dest='dropout', action='store', help="Enter the drop out decimal point", metavar="BZ", default='0.0')
    parser.add_argument("-u", "--units", dest='units', action='store', help="Enter the number of units in MLP hidden layer", metavar="BZ", default='16')
    
    parser.add_argument("-tr_coco", "--trainig_coco", dest='train_coco', action='store', help='Enter 1 if want to train with COCO rahter than UKB', metavar='COCO', default='0')
    parser.add_argument("-svemb", "--save_embeddings", dest='save_embeddings', action='store', help='Enter 1 if want to save embeddings', metavar='SVEMB', default='0')
    parser.add_argument("-pltemb", "--plot_embeddings", dest='plot_embeddings', action='store', help='Enter 1 if want to plot embeddings, note this process can take a long time and a lot of memory.', metavar='PLTEMB', default='0')
    parser.add_argument("-top_n_perc", "--top_n_perc", dest='top_n_perc', action='store', help='Enter top n percentage of snps to use. Note: if not generating pairs, it must match the dataset top n value.', metavar='TOPN', default='0.5')
    parser.add_argument("-resume", "--resume_from_batch", dest='resume_from_batch', action='store', help='Enter 1 if want to resume training from last batch checkpoint. Note: default = 0', metavar='RESUME', default='0')
    parser.add_argument("-ckpt_name", "--ckpt_name", dest='ckpt_name', action='store', help='Enter checkpoint name from batch to resume training.', metavar='ckpt_name', default='None')
    parser.add_argument("-sbp", "--subject_based_pred_flag", dest='subject_based_pred_flag', action='store', help='Enter 1 if want to train and evaluate with subject based prediction (frozen encoders).', metavar='SBP', default='0')
    parser.add_argument("-out_flag", "--out_flag", dest='out_flag', action='store', help='Enter clf for classification, reg for regression, or seq_idp for sequence and idp prediction.', metavar='OUTFLAG', default='seq_idp')
    args = parser.parse_args()

    return args 

# Main
if __name__ == '__main__':
    begin_time = time.time()
    args = parse_arguments()

    ### Set discoverable GPU cards
    # For GPU - Pytorch
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_nums #model will be trained on GPU X
    print('Current working directory: ',os.getcwd())
    ### Prepare environment ###
    if not os.path.isdir(os.path.join(os.getcwd(),'results')):
        os.mkdir(os.path.join(os.getcwd(),'results'))
        print('No result directory detected, results and checkpoints will be stored in: ',os.path.join(os.getcwd(),'results'))
    if not os.path.isdir(os.path.join(os.getcwd(),'results',args.fname_out_root)):
        os.mkdir(os.path.join(os.getcwd(),'results',args.fname_out_root))
        print(f'No results directory detected for {args.fname_out_root}, results and checkpoints will be stored in: {os.path.join(os.getcwd(),"results",args.fname_out_root)}')
    if not os.path.isfile(os.path.join(os.getcwd(),'data/pairs.pickle')):
        pairs_exist = False
        print(f'No previously made pairs detected for {args.fname_out_root}, pairs will be generated and saved in this run.')
    else:
        pairs_exist = True
        print(f'Previously made pairs detected in data directory, pairs making will be skipped and pairs will be loaded.')

    # Parse to local variables
    path_seq = args.path_seq
    path_idp = args.path_idp
    path_idp_map = args.path_idp_map
    path_pair = args.path_pairs
    path_subj_labels = args.path_subj_labels

    path_res = args.path_res
    fname_root_out = args.fname_out_root
    if args.path_ckpt == None:
        if not os.path.isdir(os.path.join(os.getcwd(),'results',args.fname_out_root,'checkpoints')):
            os.mkdir(os.path.join(os.getcwd(),'results',args.fname_out_root,'checkpoints'))
        path_ckpt = os.path.join(os.getcwd(),'results',args.fname_out_root,'checkpoints')
        print(f'No checkpoint path was given, checkpoints for {args.fname_out_root} will be stored in: {os.path.join(os.getcwd(),"results",args.fname_out_root,"checkpoints")}')
    else:
        path_ckpt = args.path_ckpt

    rnd_state = int(args.random_seed)
    tune_flag = bool(int(args.tune_flag))
    gpus_per_trial = args.gpus_per_trial

    ### Set dictionaries with paths and arguments to pass through framework ###
    paths = {
        'path_seqs' : path_seq,
        'path_idps' : path_idp,
        'path_pairs' : path_pair,
        'path_idp_map' : path_idp_map,
        'path_res' : path_res,
        'checkpoint_name' : path_ckpt,
        'tensorboard_log': os.path.join(os.getcwd(),'results',fname_root_out,'tensorboard_logs'),
        'wd' : os.getcwd(),
        'path_subj_labels' : path_subj_labels,
    }

    args = {
        'tune_flag':tune_flag,
        'gpus_per_trial':gpus_per_trial,
        'val_size' :float(int(args.val_size)/100),
        'test_size' :float(int(args.test_size)/100),
        'rnd_st' :rnd_state,
        'pairs_exist':pairs_exist,
        'fname_root_out':fname_root_out,
        # Hyperparameters
        'batch_size':int(args.batch_size),
        'learning_rate':float(args.learning_rate),
        'epochs':int(args.epochs),
        'num_layers':int(args.num_layers),
        'd_model':int(args.d_model),
        'nhead':int(args.nhead),
        'dim_feedforward':int(args.dim_feedforward),
        'dropout':float(args.dropout),
        'units':int(args.units),
        'train_coco':bool(int(args.train_coco)),
        'save_embeddings':bool(int(args.save_embeddings)),
        'plot_embeddings':bool(int(args.plot_embeddings)),
        'top_n_perc':float(args.top_n_perc),
        'resume_from_batch':bool(int(args.resume_from_batch)),
        'ckpt_name':args.ckpt_name,
        'subject_based_pred_flag':bool(int(args.subject_based_pred_flag)),
        'out_flag':args.out_flag,
    }

    # Run training, hyperparam search and testing 
    results_dict = train_eval(paths,args)
    print(f'Saving results dictionary in {os.path.join(os.getcwd(),"results",fname_root_out,"result_dict.json")}')
    with open(os.path.join(os.getcwd(),'results',fname_root_out,'result_dict.json'), "w") as outfile:
        json.dump(results_dict, outfile)
    
    print(f'Test set loss {results_dict["metrics"]["loss_test"]}')
    print(f'Test set top-1 accuracy {results_dict["metrics"]["acc_test"]}')

    # TODO: Implement function to summarize results into txt file. 
    # Results summary and return full results dictionary 
    # results_df = results_summary(results_dict, fname_root_out)
    # results_display(results_df, 'auc_test')
    # dict_path = str('../results/'+fname_root_out+'_results_dictionary.pkl')
    # quant_res_path = str('../results/'+fname_root_out+'_results_summary.csv')
                
    print("--- Total run time in %s seconds ---" % (time.time() - begin_time))





