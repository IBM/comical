import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor, from_numpy, stack, flatten,concat
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from src.utils import idp_tokenization
from collections import defaultdict
from functools import reduce

class dataset(Dataset):
    def load_seqs(self):
        self.seqs = pd.read_csv(self.path_seqs, sep='\t')
        self.seqs = self.seqs.iloc[:,1:] #added to match the new snp tokenization file (snp2vec like)
        self.seqs.loc[:,'eid'] = self.seqs.loc[:,'eid'].astype('int')
        self.seqs = self.seqs.set_index('eid')
        self.seqs_idx = self.seqs.index
        
    def load_idps(self):
        idps_raw = pd.read_csv(self.path_idps)
        self.idps_map = pd.read_csv(self.path_idp_map) # mapping file 
        idps_raw = idps_raw.loc[:, ~idps_raw.columns.str.replace("(\.\d+)$", "").duplicated()]
        idps_filt = idps_raw.dropna(thresh=len(idps_raw) - 1, axis=1) # Drop columns with all NA values on columns
        self.idps_filt = idps_filt.dropna(how= 'any', axis=0) # Drop rows (subjects) with NAs
        self.idps_filt = self.idps_filt.iloc[1:]
        self.idps_filt = self.idps_filt.set_index(self.idps_filt.iloc[:,0].astype('int'))
        # self.idps_filt = self.idps_filt.iloc[:,1:]
        self.idps_filt = self.idps_filt[~self.idps_filt.index.duplicated(keep='first')]
        self.idps_idx = self.idps_filt.index

    def load_map_and_pairs(self):
        self.idp_map = pd.read_csv(self.path_idp_map, index_col= 1)
        self.pairs = pd.read_csv(self.path_pairs)
        self.pairs.IDP = self.pairs.IDP.map(self.idp_map.Field_ID.to_dict())

    def load_subj_labels(self):
        # For regression of scores
        self.subj_labels = pd.read_csv(self.path_subj_labels)
        self.subj_labels = self.subj_labels.set_index('eid')
        self.subj_labels = self.subj_labels[~self.subj_labels.index.duplicated(keep='first')]
        # self.subj_labels = self.subj_labels.loc[:,'Age']
        # For classification of disease
        if self.disease != 'meta':
            self.subj_labels = self.subj_labels.loc[:,self.disease]
        # Do meta disease classification - i.e. if any of the diseases are present
        if self.disease == 'meta':
            self.subj_labels = self.subj_labels.loc[:,['AD','BPD','Depression','MS','Stroke','ADHD','ASD','MD','PD','SZ']]
            self.subj_labels = self.subj_labels.any(axis=1).astype('int')# collapse to one column

    def load_covariates(self):
        self.pcs = pd.read_csv(self.path_covariates)
        self.covariates = pd.read_csv(self.path_subj_labels)
        self.pcs = self.pcs.set_index('eid')
        self.covariates = self.covariates.set_index('eid')
        self.pcs = self.pcs[~self.pcs.index.duplicated(keep='first')]
        self.covariates = self.covariates[~self.covariates.index.duplicated(keep='first')]
        self.covariates = self.covariates.loc[:,['Age','Sex']]
    
    def get_iid_tensor_map(self):
        return self.iid_tensorID_map
    
    def map_iid2tensor(self):
        return pd.Index(self.matching_ids).map(self.iid_tensorID_map).values

    def load_data(self):
        self.load_seqs()
        self.load_idps()
        self.load_map_and_pairs()
        if self.subject_based_pred_flag: self.load_subj_labels()
        if self.subject_based_pred_flag: self.load_covariates()

    def match_modalities(self):
        self.matching_ids = reduce(np.intersect1d, (self.seqs_idx, self.idps_idx, self.subj_labels.index,self.covariates.index,self.pcs.index)) if self.subject_based_pred_flag else np.intersect1d(self.seqs_idx, self.idps_idx) 
        self.seqs = self.seqs.loc[self.matching_ids].sort_index()
        self.idps_filt = self.idps_filt.loc[self.matching_ids].sort_index()
        self.iid_tensorID_map = dict(zip(self.seqs.index,np.arange(len(self.seqs.index))))
        self.seqs = self.seqs.reset_index(drop=True)
        self.idps_filt = self.idps_filt.reset_index(drop=True)
        if self.subject_based_pred_flag:
            self.subj_labels = self.subj_labels.loc[self.matching_ids].sort_index()
            self.subj_labels = self.subj_labels.reset_index(drop=True)
            self.covariates = self.covariates.loc[self.matching_ids].sort_index()
            self.covariates = self.covariates.reset_index(drop=True)
            self.pcs = self.pcs.loc[self.matching_ids].sort_index()
            self.pcs = self.pcs.reset_index(drop=True)

    def set_tabular_embeddings(self):
        self.embedded_idps = idp_tokenization('cpu',64,self.rnd_st,self.idps_filt.index,{'train':self.idps_filt.values.astype('float')})
        
    def create_buckets(self):
        idps_map = pd.read_csv('data/IDPs_and_disease_mapping.csv', sep = '\t')
        snps_map = pd.read_csv('data/SNPs_and_disease_mapping_with_pvalues.csv',sep = '\t')

        self.idp_map.Field_ID = self.idp_map.Field_ID.astype('str')
        self.idps_filt.columns = self.idps_filt.columns.map({a:b for a,b in zip(self.idp_map['Field_ID'],self.idp_map.index)})
        matching_idps = np.intersect1d(self.idps_filt.columns,idps_map.index)
        idps_map = idps_map.loc[matching_idps]
        idps_map.Disease = idps_map.Disease.map({'Depression':'Unipolar_Depression',
              'Cerebrovascular_Disease':'Stroke',
              'ADHD':'ADHD',
              'PD':'PD',
              'Bipolar_Disorder':'BPD',
              'SZ':'SZ',
              'Multiple_Sclerosis':'MS',
              'Mood_Disorder':'MD',
              'Alzheimers_Disease':'AD'})

        top_n_per = self.top_n_perc
        snps_filt_pval = []
        for d in snps_map['Disease'].value_counts().index:
            snps_filt_pval.append(snps_map.loc[snps_map.loc[snps_map['Disease']==d]['P-value'].nsmallest(int(len(snps_map.loc[snps_map['Disease']==d]['P-value'])*(top_n_per/100))).index])
        snps_map = pd.concat(snps_filt_pval).loc[:, ['SNPs','Disease']]
        snps_map = snps_map.explode('Disease')
        matching_diseases = np.intersect1d(idps_map.Disease,snps_map.Disease)
        snps_map_filtered = snps_map.loc[snps_map.Disease.isin(matching_diseases)]
        idps_map_filtered = idps_map.loc[idps_map.Disease.isin(matching_diseases)]
        bucket_snps = {}
        bucket_idps = {}
        for disease in matching_diseases:
            bucket_snps[disease] = snps_map_filtered.loc[snps_map_filtered.Disease == disease].SNPs.values
            bucket_idps[disease] = idps_map_filtered.loc[idps_map_filtered.Disease == disease].index.values
        return bucket_snps, bucket_idps, matching_diseases
    
    def create_pairs(self,bucket_snps,bucket_idps,matching_diseases):
        print('Tokenization of seqs started')
        self.idp_id_map = dict(zip(self.idps_filt.columns,range(len(self.idps_filt.columns))))
        self.snp_id_map= dict(zip(self.seqs.columns,range(len(self.seqs.columns))))

        seqs_melted = self.seqs.melt()
        idps_melted = self.idps_filt.melt()
        self.embedded_idps = flatten(self.embedded_idps, start_dim=0,end_dim=1)
        pairs_all = {}
        for disease in matching_diseases:
            print(f'Creating pairs for {disease}')
            pairs_disease = {}
            for idp in tqdm(bucket_idps[disease],  desc='idp loop'):
                snps = bucket_snps[disease]
                a = seqs_melted.loc[seqs_melted.variable.isin(snps)]
                b = pd.concat([idps_melted.loc[idps_melted.variable == idp]]*len(snps)) # keeping as legacy  but probably okay to remove since we are doing embedded idps now
                a.variable, b.variable = a.variable.map(self.snp_id_map), b.variable.map(self.idp_id_map)
                pairs_disease[idp]={
                    'snp_idx':snps,
                    'idp_idx':idp,
                    'pairs' : np.concatenate((a.to_numpy(),b.to_numpy()),axis=1),
                    'embedded_idp':self.embedded_idps[idps_melted.loc[idps_melted.variable == idp].index].repeat(len(snps),1)
                 }
            pairs_all[disease] = pairs_disease

        print('Full dataset traversed, all pairs made')
        return pairs_all

    def create_pairs_for_test(self):
        self.idp_id_map = dict(zip(range(len(self.idps_filt.columns)),self.idps_filt.columns))
        self.snp_id_map= dict(zip(range(len(self.seqs.columns)),self.seqs.columns))

        pairs_all_idp = {}
        for disease in self.matching_diseases:
            print(f'Creating pairs for {disease}')
            pairs_disease = {}
            for idp in self.bucket_idps[disease]:
                snps = self.bucket_snps[disease]
                pairs_disease[idp]= snps
            pairs_all_idp[disease] = pairs_disease

            pairs_all = {}
        for disease in self.matching_diseases:
            print(f'Creating pairs for {disease}')
            pairs_disease = {}
            for snp in self.bucket_snps[disease]:
                snps = self.bucket_idps[disease]
                pairs_disease[snp]= snps
            pairs_all[disease] = pairs_disease
        
        self.dd_idp = defaultdict(list)

        for d in (pairs_all_idp['AD'],pairs_all_idp['ADHD'],pairs_all_idp['BPD'], pairs_all_idp['MD'], pairs_all_idp['MS'], pairs_all_idp['PD'], pairs_all_idp['Unipolar_Depression']): # you can list as many input dicts as you want here
            for key, value in d.items():
                self.dd_idp[key].append(value)
        self.dd = defaultdict(list)

        for d in (pairs_all['AD'],pairs_all['ADHD'],pairs_all['BPD'], pairs_all['MD'], pairs_all['MS'], pairs_all['PD'], pairs_all['Unipolar_Depression']): # you can list as many input dicts as you want here
            for key, value in d.items():
                self.dd[key].append(value)

        
    def get_pairs_for_test(self):
        return self.dd_idp, self.dd 
    
    def get_token_maps(self):
        return self.idp_id_map, self.snp_id_map

    def load_pairs(self):
        with open(os.path.join(os.getcwd(),'data/pairs.pickle'), "rb") as outfile:
            self.pair_dictionary = pickle.load(outfile)


    def extend_pairs(self):  
        self.pairs = []
        self.emb_idps = []
        print('Converting loaded dataset to Tensors started')
        for disease in tqdm(self.pair_dictionary.keys()):
            for idp in self.pair_dictionary[disease].keys():
                self.pairs.extend(self.pair_dictionary[disease][idp]['pairs'].astype('float'))
                self.emb_idps.extend(self.pair_dictionary[disease][idp]['embedded_idp'])
        self.pairs = np.asarray(self.pairs)
        self.emb_idps = stack(self.emb_idps)

    def tensorize_data(self):
        if self.subject_based_pred_flag:
            self.seqs = from_numpy(self.seqs.values)
            # self.idps_filt = from_numpy(self.idps_filt.values)
            self.target = from_numpy(self.subj_labels.values)
            self.covariates = from_numpy(self.covariates.values)
            self.pcs = from_numpy(self.pcs.values)
            # concateante covariates and pcs
            self.covariates = concat([self.covariates,self.pcs],dim=1)
        else:
            self.pairs = from_numpy(self.pairs)
    
    def reset_subject_idx_and_tensorize(self):
        # sort data frames by id
        self.seqs = self.seqs.sort_index()
        self.idps_filt = self.idps_filt.sort_index()
        # reset index
        self.seqs = self.seqs.reset_index(drop=True)
        self.idps_filt = self.idps_filt.reset_index(drop=True)
        # tensorize
        self.seqs = from_numpy(self.seqs.values)
        # self.idps_filt = from_numpy(self.idps_filt.values)
        self.target = from_numpy(self.subj_labels.values)

        
    def set_data_splits(self):
        if self.subject_based_pred_flag: # added to return subject based pairs instead of snp-idp pairs
            self.train_idx, self.test_idx, _, _ = train_test_split(
             np.arange(len(self.seqs)), np.zeros(len(self.seqs)), test_size=self.test_size, random_state=self.rnd_st)

            self.train_idx, self.val_idx, _, _  = train_test_split(
                self.train_idx, np.zeros(len(self.train_idx)), test_size = self.val_size, random_state=self.rnd_st)
        else:
            self.train_idx, self.test_idx, _, _ = train_test_split(
                np.arange(len(self.pairs)), np.zeros(len(self.pairs)), test_size=self.test_size, random_state=self.rnd_st)

            self.train_idx, self.val_idx, _, _  = train_test_split(
                self.train_idx, np.zeros(len(self.train_idx)), test_size = self.val_size, random_state=self.rnd_st) 
    
    def get_data_splits(self):
        return self.train_idx, self.val_idx, self.test_idx
    
    def get_labels(self):
        return self.subj_labels

    def __init__(self,paths, args):
        # Set paths and args variables
        self.path_seqs, self.path_idps, self.path_idp_map, self.path_pairs = paths['path_seqs'], paths['path_idps'], paths['path_idp_map'], paths['path_pairs']
        self.val_size, self.test_size, self.rnd_st, self.pairs_exist, self.fname_root_out, self.top_n_perc = args['val_size'], args['test_size'], args['rnd_st'], args['pairs_exist'], args['fname_root_out'], args['top_n_perc']
        self.subject_based_pred_flag = args['subject_based_pred_flag'] # added to return subject based pairs instead of snp-idp pairs
        self.path_subj_labels = paths['path_subj_labels']
        self.path_covariates = paths['path_covariates']
        self.disease = args['target']

        self.load_data()
        self.match_modalities()
        self.bucket_snps, self.bucket_idps, self.matching_diseases = self.create_buckets()

        if self.subject_based_pred_flag: # added to return subject based pairs instead of snp-idp pairs
            # self.load_subj_labels()
            # self.load_covariates()
            # self.reset_subject_idx_and_tensorize()
            self.set_tabular_embeddings()
            self.set_data_splits()
            self.tensorize_data()
        else:
            if self.pairs_exist:
                self.load_pairs()
            else:
                self.set_tabular_embeddings()# remains to be decided where to perform as this will require to first perfrom the data splits
                self.pair_dictionary = self.create_pairs(self.bucket_snps, self.bucket_idps, self.matching_diseases)
                print(f'Saving created pairs in {os.path.join(os.getcwd(),"data/pairs.pickle")}')
                with open(os.path.join(os.getcwd(),'data/pairs.pickle'), "wb") as outfile:
                    pickle.dump(self.pair_dictionary, outfile)

            # self.tensorize_data()
            self.extend_pairs()
            self.set_data_splits()
            self.create_pairs_for_test()
    
    def __len__(self):
        # TODO: Update based on data pair definition
        return len(self.pairs)

    def __getitem__(self, index):
        if self.subject_based_pred_flag: # added to return subject based pairs instead of snp-idp pairs
            return np.arange(self.seqs.shape[1]), self.seqs[index], np.arange(len(self.idps_filt.columns)), self.embedded_idps[index], self.target[index], self.covariates[index]
        else:
            return self.pairs[index,0], self.pairs[index,1], self.pairs[index,2], self.emb_idps[index], self.pairs[index,0], self.pairs[index,0] # last 2 elements are added to match the return format of the subject based pairs, can be regarded as none
