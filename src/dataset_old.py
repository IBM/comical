import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor, from_numpy, stack, flatten
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle
from src.utils import idp_tokenization
from collections import defaultdict

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
        # self.idp_map.index = self.idp_map.index.str.replace('(', '')
        # self.idp_map.index = self.idp_map.index.str.replace('hemisphere\)', '')
        # self.idp_map.index = self.idp_map.index.str.replace('\)', '')
        self.pairs = pd.read_csv(self.path_pairs)
        self.pairs.IDP = self.pairs.IDP.map(self.idp_map.Field_ID.to_dict())

    
    def get_iid_tensor_map(self):
        return self.iid_tensorID_map
    
    def map_iid2tensor(self):
        return pd.Index(self.matching_ids).map(self.iid_tensorID_map).values

    def load_data(self):
        self.load_seqs()
        self.load_idps()
        self.load_map_and_pairs()

    def match_modalities(self):
        self.matching_ids = np.intersect1d(self.seqs_idx, self.idps_idx)
        self.seqs = self.seqs.loc[self.matching_ids].sort_index()
        self.idps_filt = self.idps_filt.loc[self.matching_ids].sort_index()
        self.iid_tensorID_map = dict(zip(self.seqs.index,np.arange(len(self.seqs.index))))
        self.seqs = self.seqs.reset_index(drop=True)
        self.idps_filt = self.idps_filt.reset_index(drop=True)

    def tokenize_seqs(self,sequences):
        # Import the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-1000g")

        # Create a dummy dna sequence and tokenize it
        # replaces these sequences with those stored in 'neuroMRI_qced_seqs.csv'
        tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt")["input_ids"]
        return tokens_ids[:,1]
    
    def tokenize_snps(self,sequences):
        snps_dict = {"A": "A", "G": "G", "T": "T", "C": "C", "N": "N", "DEL": "X", "AI": "B", "GI": "H", "TI": "U", "CI": "D", "NI": "O", "A_A": "啊", "A_G": "阿", "A_T": "锕", "A_C": "嗄", "A_N": "呵", "A_DEL": "腌", "A_AI": "吖", "A_GI": "安", "A_TI": "按", "A_CI": "俺", "A_NI": "案", "G_A": "哥", "G_G": "跟", "G_T": "个", "G_C": "糕", "G_N": "给", "G_DEL": "过", "G_AI": "更", "G_GI": "高", "G_TI": "该", "G_CI": "古", "G_NI": "股", "T_A": "他", "T_G": "她", "T_T": "它", "T_C": "塔", "T_N": "踏", "T_DEL": "塌", "T_AI": "嗒", "T_GI": "挞", "T_TI": "拓", "T_CI": "特", "T_NI": "铽", "C_A": "擦", "C_G": "嚓", "C_T": "礤", "C_C": "囃", "C_N": "拆", "C_DEL": "吃", "C_AI": "成", "C_GI": "床", "C_TI": "出", "C_CI": "车", "C_NI": "穿", "N_A": "那", "N_G": "拿", "N_T": "哪", "N_C": "纳", "N_N": "呐", "N_DEL": "钠", "N_AI": "捺", "N_GI": "乸", "N_TI": "娜", "N_CI": "呢", "N_NI": "讷", "DEL_A": "谢", "DEL_G": "想", "DEL_T": "徐", "DEL_C": "小", "DEL_N": "学", "DEL_DEL": "下", "DEL_AI": "新", "DEL_GI": "香", "DEL_TI": "需", "DEL_CI": "许", "DEL_NI": "虚", "AI_A": "吧", "AI_G": "八", "AI_T": "把", "AI_C": "爸", "AI_N": "巴", "AI_DEL": "拔", "AI_AI": "扒", "AI_GI": "霸", "AI_TI": "笔", "AI_CI": "比", "AI_NI": "逼", "GI_A": "哈", "GI_G": "蛤", "GI_T": "虾", "GI_C": "铪", "GI_N": "蝦", "GI_DEL": "鉿", "GI_AI": "和", "GI_GI": "喝", "GI_TI": "盒", "GI_CI": "何", "GI_NI": "合", "TI_A": "我", "TI_G": "为", "TI_T": "万", "TI_C": "完", "TI_N": "玩", "TI_DEL": "无", "TI_AI": "问", "TI_GI": "位", "TI_TI": "哇", "TI_CI": "娃", "TI_NI": "瓦", "CI_A": "大", "CI_G": "懂", "CI_T": "瘩", "CI_C": "打", "CI_N": "达", "CI_DEL": "搭", "CI_AI": "哒", "CI_GI": "答", "CI_TI": "沓", "CI_CI": "地", "CI_NI": "第", "NI_A": "哦", "NI_G": "噢", "NI_T": "喔", "NI_C": "筽", "NI_N": "毮", "NI_DEL": "弄", "NI_AI": "浓", "NI_GI": "农", "NI_TI": "侬", "NI_CI": "脓", "NI_NI": "哝"}
        # map loaded subject snps using snp dict
        self.seqs = self.seqs.map(snps_dict)


    def set_tabular_embeddings(self):
        self.embedded_idps = idp_tokenization('cpu',64,self.rnd_st,self.idps_filt.index,{'train':self.idps_filt.values.astype('float')})
        
    def create_buckets(self):
        idps_map = pd.read_csv('data/IDPs_and_disease_mapping.csv', sep = '\t')
        snps_map = pd.read_csv('data/SNPs_and_disease_mapping_with_pvalues.csv',sep = '\t')
        # idps_map.index = idps_map.index.str.replace('_', ' ')
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
        # snps_map = pd.read_csv('data/SNPs_and_disease_mapping.csv', sep = '\t')
        # snps_map.Disease = snps_map.Disease.apply(lambda x: x.split(', '))
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
        # batch_size = len(matching_diseases)
        # batch = np.empty((batch_size,batch_size),dtype='str')
        # for b in range(batch_size):
        #     for i,disease in enumerate(matching_diseases_2):
        #         batch[b,i] = np.random.choice(bucket_snps[disease],size=1)[0]
        #         batch[i,b] = np.random.choice(bucket_idps[disease],size=1)[0]
        return bucket_snps, bucket_idps, matching_diseases
    
    def create_pairs(self,bucket_snps,bucket_idps,matching_diseases):
        print('Tokenization of seqs started')
        self.idp_id_map = dict(zip(self.idps_filt.columns,range(len(self.idps_filt.columns))))
        self.snp_id_map= dict(zip(self.seqs.columns,range(len(self.seqs.columns))))
        # seq_map  = dict(zip(np.unique(self.seqs.values.flatten()),range(len(np.unique(self.seqs.values.flatten())))))
        # pairs = []

        # for b in tqdm(range(10000)):
        #     for i,disease in enumerate(matching_diseases):
        #         snp_name = np.random.choice(bucket_snps[disease],size=1)[0]
        #         idp_name = np.random.choice(bucket_idps[disease],size=1)[0]
        #         pairs.extend((self.idps_filt.loc[b,idp_name], self.tokenize_seqs(self.seqs.loc[b,snp_name])[0]))
        # for disease in matching_diseases:
        seqs_melted = self.seqs.melt()
        idps_melted = self.idps_filt.melt()
        self.embedded_idps = flatten(self.embedded_idps, start_dim=0,end_dim=1)
        pairs_all = {}
        for disease in matching_diseases:
            print(f'Creating pairs for {disease}')
            pairs_disease = {}
            for idp in tqdm(bucket_idps[disease][:10],  desc='idp loop'):
            # for idp in tqdm(bucket_idps[disease],  desc='idp loop'):
                snps = bucket_snps[disease]
                a = seqs_melted.loc[seqs_melted.variable.isin(snps)]
                b = pd.concat([idps_melted.loc[idps_melted.variable == idp]]*len(snps)) # keeping as legacy  but probably okay to remove since we are doing embedded idps now
                a.variable, b.variable = a.variable.map(self.snp_id_map), b.variable.map(self.idp_id_map)
                # a.value = self.tokenize_seqs(a.value)
                # a.value = a.value.map(seq_map)
                pairs_disease[idp]={
                    'snp_idx':snps,
                    'idp_idx':idp,
                    'pairs' : np.concatenate((a.to_numpy(),b.to_numpy()),axis=1),
                    'embedded_idp':self.embedded_idps[idps_melted.loc[idps_melted.variable == idp].index].repeat(len(snps),1)
                 }
            pairs_all[disease] = pairs_disease
                # for snp in tqdm(bucket_snps[disease], desc='snp loop'):
                #     pairs.extend(zip(self.idps_filt.loc[:,idp], self.tokenize_seqs(self.seqs.loc[:,snp]).numpy(),[disease]*len(self.idps_filt)))
        
        print('Full dataset traversed, all pairs made')
        return pairs_all

    # def create_pairs(self):
    #     # two possible ways to approach it.
    #     # If we follow CLIP, we could define pairs of IDPS and SNPs following the GWAS-IDPs paper -> CE loss predicting to what IDP the SNP belongs and vice-versa.
    #     # If we follow a traditional contrastive framework, we apply a distance loss between the pairs -> same patient being a pair.
        
    #     # Select relevant columns from IDP and Seqs
    #     ad_idps = self.idps_filt.loc[:,self.idps_filt.columns.isin(self.pairs.IDP.astype('str'))]
    #     ad_seqs = self.seqs.loc[:,self.seqs.columns.isin(self.pairs.SNP)]
    #     ad_seqs = self.seqs.iloc[:,:99]
    #     notad_idps = self.idps_filt.loc[:,~self.idps_filt.columns.isin(self.pairs.IDP.astype('str'))]
    #     notad_seqs = self.seqs.loc[:,~self.seqs.columns.isin(self.pairs.SNP)]
    #     # 
    #     pairs = []
    #     # num_batches = self.seqs_filt.shape[0] / args['batch_size']

    #     print('Tokenization of seqs started')
    #     # for i in tqdm(range(ad_idps.shape[1])):
    #     for i in tqdm(range(3)):
    #         # Randomly sample n rows
    #         # sampled_idx = self.idps_filt.sample(n=args['batch_size']).index
    #         # Get the IDP and Seqs corresponding to the randomly sampled idx
    #         pairs.extend(zip(ad_idps.iloc[:,i].values,self.tokenize_seqs(ad_seqs.iloc[:,i].values)))
    #         pairs.extend(zip(notad_idps.iloc[:,i].values,self.tokenize_seqs(notad_seqs.iloc[:,i].values)))
    #     print('Tokenization of seqs completed')
    #     return pairs

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
        # self.seqs = from_numpy(self.seqs.values) # cannot tensorize sequences, that will have to be done in the data loder. Or tokenize first here
        # self.pairs = from_numpy(np.asarray(data, dtype = 'float').reshape(-1,2))
        # self.pairs = from_numpy((np.asarray(data)[:,0:2]).astype('float'))
        # snp_id_map = dict(zip(self.idps_filt.columns,range(len(self.idps_filt.columns))))
        # idp_id_map = dict(zip(self.seqs.columns,range(len(self.seqs.columns))))
        
        
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
        self.pairs = from_numpy(self.pairs)
        
    def set_data_splits(self):
        self.train_idx, self.test_idx, _, _ = train_test_split(
             np.arange(len(self.pairs)), np.zeros(len(self.pairs)), test_size=self.test_size, random_state=self.rnd_st)

        self.train_idx, self.val_idx, _, _  = train_test_split(
            self.train_idx, np.zeros(len(self.train_idx)), test_size = self.val_size, random_state=self.rnd_st) 
    
    def get_data_splits(self):
        return self.train_idx, self.val_idx, self.test_idx

    def __init__(self,paths, args):
        # Set paths and args variables
        self.path_seqs, self.path_idps, self.path_idp_map, self.path_pairs = paths['path_seqs'], paths['path_idps'], paths['path_idp_map'], paths['path_pairs']
        self.val_size, self.test_size, self.rnd_st, self.pairs_exist, self.fname_root_out, self.top_n_perc = args['val_size'], args['test_size'], args['rnd_st'], args['pairs_exist'], args['fname_root_out'], args['top_n_perc']

        self.load_data()
        self.match_modalities()
        self.bucket_snps, self.bucket_idps, self.matching_diseases = self.create_buckets()
        
        if self.pairs_exist:
            self.load_pairs()
        else:
            # self.load_data()
            # self.match_modalities()
            self.set_tabular_embeddings()# remains to be decided where to perform as this will require to first perfrom the data splits
            # self.tokenize_seqs()
            # self.bucket_snps, self.bucket_idps, self.matching_diseases = self.create_buckets()
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
        # return self.pairs[index,0], self.pairs[index,1], self.pairs[index,2], self.pairs[index,3]
        return self.pairs[index,0], self.pairs[index,1], self.pairs[index,2], self.emb_idps[index]
        # return self.pairs[index,0], self.pairs[index,1]#, self.seqs[index], self.tokens_ids
    


#%%
# paths = {
#     'path_seqs' : '~/ukb-pgx/comical/comical/data/neuroMRI_qced_seqs.csv',
#     'path_idps' : '~/ukb-pgx/comical/comical/data/T1_struct_brainMRI_IDPs.csv',
#     'path_idp_map' : '~/ukb-pgx/comical/comical/data/T1mri.csv',
#     'path_pairs' : '~/ukb-pgx/comical/comical/data/pairs.csv',
# }
# args = {
#     'val_size':0.20,
#     'test_size':0.10,
#     'rnd_st':0
# }

# data = dataset(paths, args)
# train_idx, val_idx, test_idx = data.get_data_splits()
# print(data[test_idx].shape)
# assert np.intersect1d(train_idx, val_idx).size == 0 and np.intersect1d(train_idx, test_idx).size == 0 and np.intersect1d(val_idx, test_idx).size == 0