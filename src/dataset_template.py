import os, math
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
    def load_modality(self, path, index_col = 0):
        # TODO: Delete if path == self.path_mod_a else for final version, just used for debugging
        if path == self.path_mod_a:
            mod = pd.read_csv(path, sep='\t')
            mod = mod.iloc[:,1:] #added to match the new snp tokenization file (snp2vec like)
            if type(index_col) == int:
                mod = mod.set_index(mod.iloc[:,index_col].astype('int'))
            else:
                mod = mod.set_index(mod.loc[:,index_col].astype('int'))
        else:
            idps_raw = pd.read_csv(path)
            idps_raw = idps_raw.loc[:, ~idps_raw.columns.str.replace("(\.\d+)$", "").duplicated()]
            idps_filt = idps_raw.dropna(thresh=len(idps_raw) - 1, axis=1) # Drop columns with all NA values on columns
            self.idps_filt = idps_filt.dropna(how= 'any', axis=0) # Drop rows (subjects) with NAs
            self.idps_filt = self.idps_filt.iloc[1:]
            self.idps_filt = self.idps_filt.set_index(self.idps_filt.iloc[:,0].astype('int'))
            # self.idps_filt = self.idps_filt.iloc[:,1:]
            self.idps_filt = self.idps_filt[~self.idps_filt.index.duplicated(keep='first')]
            self.idps_idx = self.idps_filt.index
            return self.idps_filt, self.idps_idx
        return mod, mod.index
        

    def load_colname_mapping(self,mod_b_colname):
        # Can be removed if we are not using the map - template shouldn't use it as data should be properly formatted
        if self.path_mod_b_map:
            self.mod_b_map = pd.read_csv(self.path_mod_b_map, index_col= 1)
            self.mod_b_map['Field_ID'] = self.mod_b_map['Field_ID'].astype('str')

    def load_target_labels(self, index_col = 0):
        # For regression of scores
        target_labels = pd.read_csv(self.path_target_labels)
        target_labels = target_labels.set_index(index_col)
        # Remove duplicates if they exist
        if target_labels.index.duplicated().any():
            target_labels = target_labels[~target_labels.index.duplicated(keep='first')]
                
        # For classification of target in downstream task
        target_labels = target_labels.loc[:,self.grouping]
        return target_labels

    def load_covariates(self, covariates_names, index_col = 0):
        if self.path_covariates:
            self.pcs = pd.read_csv(self.path_covariates)
            self.pcs = self.pcs.set_index(index_col)
            self.pcs = self.pcs[~self.pcs.index.duplicated(keep='first')]
        
        self.covariates = pd.read_csv(self.path_target_labels)
        self.covariates = self.covariates.set_index(index_col)
        self.covariates = self.covariates[~self.covariates.index.duplicated(keep='first')]
        self.covariates = self.covariates.loc[:,covariates_names]
    
    def load_idx_subset(self, path):
        idx = np.loadtxt(path, delimiter=',', dtype=int)
        return idx.astype('int')
    
    def get_iid_tensor_map(self):
        return self.iid_tensorID_map
    
    def map_iid2tensor(self):
        return pd.Index(self.matching_ids).map(self.iid_tensorID_map).values

    def load_data(self, index_col, covariates_names = None):
        self.mod_a, self.mod_a_idx = self.load_modality(self.path_mod_a ,index_col)
        self.mod_b, self.mod_b_idx = self.load_modality(self.path_mod_b, index_col)
        self.load_colname_mapping('IDP')
        if self.downstream_pred_task_flag:
            self.target_labels = self.load_target_labels(index_col)
            self.load_covariates(covariates_names, index_col)
            if self.task == 1:
                idx = self.load_idx_subset(self.path_subset_idx)
                # Filter the data based on the idx
                self.mod_a, self.mod_a_idx = self.mod_a.loc[self.mod_a_idx.intersection(idx)], self.mod_a_idx.intersection(idx)
                self.mod_b, self.mod_b_idx = self.mod_b.loc[self.mod_b_idx.intersection(idx)], self.mod_b_idx.intersection(idx)
            if self.task == 2:
                self.mix_idx = self.load_idx_subset(self.path_subset_idx)
                self.hc_idx = self.load_idx_subset(self.path_subset_idx_unseen)
                # Filter the data based on the union of mix_idx and hc_idx.shape

                self.mod_a, self.mod_a_idx = self.mod_a.loc[np.union1d(self.mix_idx, self.hc_idx)], self.mod_a_idx.intersection(np.union1d(self.mix_idx, self.hc_idx))
                self.mod_b, self.mod_b_idx = self.mod_b.loc[np.union1d(self.mix_idx, self.hc_idx)], self.mod_b_idx.intersection(np.union1d(self.mix_idx, self.hc_idx))



        if self.specific_idx_pair_creation_flag:
            idx = self.load_idx_subset(self.path_subset_idx)
            # Filter the data based on the idx
            self.mod_a, self.mod_a_idx = self.mod_a.loc[self.mod_a_idx.intersection(idx)], self.mod_a_idx.intersection(idx)
            self.mod_b, self.mod_b_idx = self.mod_b.loc[self.mod_b_idx.intersection(idx)], self.mod_b_idx.intersection(idx)

    def match_modalities(self):
        self.matching_ids = reduce(np.intersect1d, (self.mod_a_idx, self.mod_b_idx, self.target_labels.index,self.covariates.index,self.pcs.index)) if self.downstream_pred_task_flag else np.intersect1d(self.mod_a_idx, self.mod_b_idx) 
        self.mod_a = self.mod_a.loc[self.matching_ids].sort_index()
        self.mod_b = self.mod_b.loc[self.matching_ids].sort_index()
        self.iid_tensorID_map = dict(zip(self.mod_a.index,np.arange(len(self.mod_a.index))))
        self.mod_a = self.mod_a.reset_index(drop=True)
        self.mod_b = self.mod_b.reset_index(drop=True)
        if self.downstream_pred_task_flag:
            self.target_labels = self.target_labels.loc[self.matching_ids].sort_index()
            self.target_labels = self.target_labels.reset_index(drop=True)
            self.covariates = self.covariates.loc[self.matching_ids].sort_index()
            self.covariates = self.covariates.reset_index(drop=True)
            self.pcs = self.pcs.loc[self.matching_ids].sort_index()
            self.pcs = self.pcs.reset_index(drop=True)
            # Added to deal with PRS smaller set of ids
            if self.task == 2:
                self.mix_idx = np.intersect1d(self.mix_idx, self.matching_ids)
                self.hc_idx = np.intersect1d(self.hc_idx, self.matching_ids)

    def set_tabular_embeddings(self, count, data):
        return idp_tokenization('cpu',count,self.rnd_st,data.index,{'train':data.values.astype('float')})
        
    def create_buckets(self, top_n_per, feat_a_target_col = None, feat_b_target_col = None, feat_a_index_col = None, feat_b_index_col = None):
        # load the maps
        feats_a_map = pd.read_csv(self.path_mod_a2group_map,sep = '\t')
        feats_b_map = pd.read_csv(self.path_mod_b2group_map, sep = '\t')
        # set indices
        feat_a_index_col = feat_a_index_col if feat_a_index_col else feats_a_map.index
        feat_b_index_col = feat_b_index_col if feat_b_index_col else feats_b_map.index

        # TODO: Delete this map for final version, not necessary for template
        self.mod_b.columns = self.mod_b.columns.map({a:b for a,b in zip(self.mod_b_map['Field_ID'],self.mod_b_map.index)})
        # Make sure that all the features in the map are present in the data
        feats_b_map = feats_b_map.loc[np.intersect1d(self.mod_b.columns,feats_b_map.index)]
        # TODO: Delete this map for final version
        feats_b_map[feat_b_target_col] = feats_b_map[feat_b_target_col].map({'Depression':'Unipolar_Depression',
              'Cerebrovascular_Disease':'Stroke',
              'ADHD':'ADHD',
              'PD':'PD',
              'Bipolar_Disorder':'BPD',
              'SZ':'SZ',
              'Multiple_Sclerosis':'MS',
              'Mood_Disorder':'MD',
              'Alzheimers_Disease':'AD'})

        snps_filt_pval = []
        for d in feats_a_map[feat_a_target_col].value_counts().index:
            # snps_filt_pval.append(feats_a_map.loc[feats_a_map.loc[feats_a_map[feat_a_target_col]==d]['P-value'].nsmallest(int(len(feats_a_map.loc[feats_a_map[feat_a_target_col]==d]['P-value'])*(top_n_per/100))).index])
            # Updated the line above to make sure that at least 1 snp is selected. New version allows to select 1 snp if the number of snps matching theis less than 1
            snps_filt_pval.append(feats_a_map.loc[feats_a_map.loc[feats_a_map[feat_a_target_col] == d, 'P-value'].nsmallest(max(1, int(len(feats_a_map.loc[feats_a_map[feat_a_target_col] == d, 'P-value']) * (top_n_per / 100)))).index])
        feats_a_map = pd.concat(snps_filt_pval).loc[:, [feat_a_index_col,feat_a_target_col]]
        feats_a_map = feats_a_map.explode(feat_a_target_col)
        matching_grouping = np.intersect1d(feats_b_map[feat_b_target_col],feats_a_map[feat_a_target_col])
        feats_a_map_filtered = feats_a_map.loc[feats_a_map[feat_a_target_col].isin(matching_grouping)]
        feats_b_map_filtered = feats_b_map.loc[feats_b_map[feat_b_target_col].isin(matching_grouping)]
        bucket_mod_a = {}
        bucket_mod_b = {}
        for grouping in matching_grouping:
            bucket_mod_a[grouping] = feats_a_map_filtered.loc[feats_a_map_filtered[feat_a_target_col] == grouping].SNPs.values
            bucket_mod_b[grouping] = feats_b_map_filtered.loc[feats_b_map_filtered[feat_b_target_col] == grouping].index.values
        return bucket_mod_a, bucket_mod_b, matching_grouping
    
    def create_pairs(self,bucket_mod_a,bucket_mod_b,matching_grouping):
        print('Tokenization of modality started')
        self.mod_b_id_map = dict(zip(self.mod_b.columns,range(len(self.mod_b.columns))))
        self.mod_a_id_map= dict(zip(self.mod_a.columns,range(len(self.mod_a.columns))))

        mod_a_melted = self.mod_a.melt()
        mod_b_melted = self.mod_b.melt()
        self.embedded_data = flatten(self.embedded_data, start_dim=0,end_dim=1)
        pairs_all = {}
        for grouping in matching_grouping:
            print(f'Creating pairs for {grouping}')
            pairs_grouping = {}
            for mod_b_item in tqdm(bucket_mod_b[grouping],  desc='Modality B loop'):
                mod_a_item = bucket_mod_a[grouping]
                a = mod_a_melted.loc[mod_a_melted.variable.isin(mod_a_item)]
                b = pd.concat([mod_b_melted.loc[mod_b_melted.variable == mod_b_item]]*len(mod_a_item)) # keeping as legacy  but probably okay to remove since we are doing embedded idps now
                a.variable, b.variable = a.variable.map(self.mod_a_id_map), b.variable.map(self.mod_b_id_map)
                pairs_grouping[mod_b_item]={
                    'mod_a_item_idx':mod_a_item,
                    'mod_b_item_idx':mod_b_item,
                    'pairs' : np.concatenate((a.to_numpy(),b.to_numpy()),axis=1),
                    'embedded_mod_b_item':self.embedded_data[mod_b_melted.loc[mod_b_melted.variable == mod_b_item].index].repeat(len(mod_a_item),1)
                 }
            pairs_all[grouping] = pairs_grouping

        print('Full dataset traversed, all pairs made')
        return pairs_all

    def create_pairs_for_test(self):
        self.mod_b_id_map = dict(zip(range(len(self.mod_b.columns)),self.mod_b.columns))
        self.mod_a_id_map= dict(zip(range(len(self.mod_a.columns)),self.mod_a.columns))

        pairs_all_mod_b = {}
        for grouping in self.matching_grouping:
            print(f'Creating pairs for {grouping}')
            pairs_grouping = {}
            for mod_b_item in self.bucket_mod_b[grouping]:
                mod_a_items = self.bucket_mod_a[grouping]
                pairs_grouping[mod_b_item]= mod_a_items
            pairs_all_mod_b[grouping] = pairs_grouping

        pairs_all_mod_a= {}
        for grouping in self.matching_grouping:
            print(f'Creating pairs for {grouping}')
            pairs_grouping = {}
            for mod_a_item in self.bucket_mod_a[grouping]:
                mod_b_items = self.bucket_mod_b[grouping]
                pairs_grouping[mod_a_item]= mod_b_items
            pairs_all_mod_a[grouping] = pairs_grouping

        # For every key in the dictionary, get the values and append them to a list
        dd_mod_a = defaultdict(list)
        for k in pairs_all_mod_a.keys():
            for key, value in pairs_all_mod_a[k].items():
                dd_mod_a[key].append(value)

        dd_mod_b = defaultdict(list)
        for k in pairs_all_mod_a.keys():
            for key, value in pairs_all_mod_b[k].items():
                dd_mod_b[key].append(value)
        
        return dd_mod_a, dd_mod_b

        
    def get_pairs_for_test(self):
        return self.dd_mod_b, self.dd_mod_a
    
    def get_token_maps(self):
        return self.mod_b_id_map, self.mod_a_id_map

    def load_pairs(self,path_saved_pairs):
        with open(path_saved_pairs, "rb") as outfile:
            self.pair_dictionary = pickle.load(outfile)

    def get_disease_idp_snp_map(self):
        feats_a_map = pd.read_csv(self.path_mod_a2group_map,sep = '\t')
        feats_b_map = pd.read_csv(self.path_mod_b2group_map, sep = '\t')
        return feats_a_map, feats_b_map


    def extend_pairs(self):  
        self.pairs = []
        self.mod_b_items = []
        print('Converting loaded dataset to Tensors started')
        for grouping in tqdm(self.pair_dictionary.keys()):
            for mod_b_item in self.pair_dictionary[grouping].keys():
                self.pairs.extend(self.pair_dictionary[grouping][mod_b_item]['pairs'].astype('float'))
                self.mod_b_items.extend(self.pair_dictionary[grouping][mod_b_item]['embedded_mod_b_item']) # embedded_mod_b_item | embedded_idp
        self.pairs = np.asarray(self.pairs)
        self.mod_b_items = stack(self.mod_b_items)

    def tensorize_data(self):
        if self.downstream_pred_task_flag:
            self.mod_a = from_numpy(self.mod_a.drop('eid', axis =1).values)
            self.target = from_numpy(self.target_labels.values)
            self.covariates = from_numpy(self.covariates.values)
            self.pcs = from_numpy(self.pcs.values)
            # concateante covariates and pcs
            self.covariates = concat([self.covariates,self.pcs],dim=1)
        else:
            self.pairs = from_numpy(self.pairs)
    
    def reset_subject_idx_and_tensorize(self):
        # sort data frames by id
        self.mod_a = self.mod_a.sort_index()
        self.mod_b = self.mod_b.sort_index()
        # reset index
        self.mod_a = self.mod_a.reset_index(drop=True)
        self.mod_b = self.mod_b.reset_index(drop=True)
        # tensorize
        self.mod_a = from_numpy(self.mod_a.values)
        self.target = from_numpy(self.target_labels.values)

        
    def set_data_splits(self):
        if self.downstream_pred_task_flag: # added to return subject based pairs instead of snp-idp pairs
            if self.task == 1: # Classify patients postitive vs controls - stratified split
                self.train_idx, self.test_idx, _, _ = train_test_split(
                    np.arange(len(self.mod_a)), self.target_labels, test_size=self.test_size, random_state=self.rnd_st, stratify=self.target_labels)
                self.train_idx, self.val_idx, _, _  = train_test_split(
                    self.train_idx, self.target_labels[self.train_idx], test_size = self.val_size, random_state=self.rnd_st, stratify=self.target_labels[self.train_idx])

            elif self.task == 2: # Regression of pseudo prs - train on D/HC mix an evalaute on HC only
                idx_map = dict(zip(np.union1d(self.mix_idx, self.hc_idx),range(len(self.mod_a_idx))))
                # train val split using the mix idx only, use the idx_map to map the indices from the original indices to the new indices
                self.train_idx, self.val_idx, _, _  = train_test_split(
                    np.array([idx_map[i] for i in self.mix_idx]), self.target_labels[np.array([idx_map[i] for i in self.mix_idx])], test_size = self.val_size, random_state=self.rnd_st
                )
                self.test_idx = np.array([idx_map[i] for i in self.hc_idx])
                
            else:
                self.train_idx, self.test_idx, _, _ = train_test_split(
                np.arange(len(self.mod_a)), np.zeros(len(self.mod_a)), test_size=self.test_size, random_state=self.rnd_st)

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
        return self.target_labels

    def __init__(self,paths, args):
        # Set paths and args variables
        # Necessary paths
        self.path_mod_a, self.path_mod_b, self.path_mod_b_map = paths['path_mod_a'], paths['path_mod_b'], paths['path_mod_b_map']
        self.val_size, self.test_size, self.rnd_st, self.pairs_exist, self.fname_root_out, self.top_n_perc = args['val_size'], args['test_size'], args['rnd_st'], args['pairs_exist'], args['fname_root_out'], args['top_n_perc']
        self.downstream_pred_task_flag = args['downstream_pred_task_flag']
        self.path_target_labels = paths['path_target_labels']
        self.path_covariates = paths['path_covariates']
        self.path_mod_a2group_map = paths['path_mod_a2group_map']
        self.path_mod_b2group_map = paths['path_mod_b2group_map']
        path_saved_pairs = paths['path_saved_pairs']
        self.path_data = paths["path_data"]
        self.grouping = args['target']
        index_col = args['index_col']
        covariates_names =  args['covariates_names']
        count_bins = args['count_bins']
        feat_a_target_col = args['feat_a_target_col']
        feat_b_target_col = args['feat_b_target_col']
        feat_a_index_col = args['feat_a_index_col']
        feat_b_index_col = args['feat_b_index_col']
        self.specific_idx_pair_creation_flag = args['sub_idx_flag']
        self.path_subset_idx = paths['path_sub_idx']
        self.path_subset_idx_unseen = paths['path_sub_idx_unseen']
        self.task = 1 if args['out_flag'] == 'clf' else 2 if args['out_flag'] == 'reg' else 0

        self.load_data(index_col = index_col, covariates_names = covariates_names)
        self.match_modalities()
        self.bucket_mod_a, self.bucket_mod_b, self.matching_grouping = self.create_buckets(top_n_per = self.top_n_perc, feat_a_target_col=feat_a_target_col,feat_b_target_col = feat_b_target_col , feat_a_index_col = feat_a_index_col, feat_b_index_col= feat_b_index_col)

        if self.downstream_pred_task_flag: # added to return subject based pairs instead of snp-idp pairs
            # self.load_target_labels()
            # self.load_covariates()
            # self.reset_subject_idx_and_tensorize()
            self.embedded_data = self.set_tabular_embeddings(count_bins, self.mod_b)
            self.set_data_splits()
            self.tensorize_data()
        else:
            if self.pairs_exist:
                self.load_pairs(path_saved_pairs)
            else:
                self.embedded_data = self.set_tabular_embeddings(count_bins, self.mod_b)# remains to be decided where to perform as this will require to first perfrom the data splits
                self.pair_dictionary = self.create_pairs(self.bucket_mod_a, self.bucket_mod_b, self.matching_grouping)
                print(f'Saving created pairs in {path_saved_pairs}')
                with open(path_saved_pairs, "wb") as outfile:
                    pickle.dump(self.pair_dictionary, outfile)
            # self.tensorize_data()
            self.extend_pairs()
            self.set_data_splits()
            self.dd_mod_a, self.dd_mod_b = self.create_pairs_for_test()
    
    def __len__(self):
        # TODO: Update based on data pair definition
        return len(self.pairs)

    def __getitem__(self, index):
        # Output format input_a_id,seq_a,input_b_id,seq_b,target,covariates
        if self.downstream_pred_task_flag: # added to return subject based pairs instead of snp-idp pairs
            return np.arange(self.mod_a.shape[1]), self.mod_a[index], np.arange(len(self.mod_b.columns)), self.embedded_data[index], self.target[index], self.covariates[index]
        else:
            return self.pairs[index,0], self.pairs[index,1], self.pairs[index,2], self.mod_b_items[index], self.pairs[index,0], self.pairs[index,0] # last 2 elements are added to match the return format of the subject based pairs, can be regarded as none
