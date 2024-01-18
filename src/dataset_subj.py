import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor, from_numpy,vstack
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
# from utils import 

class dataset(Dataset):
    def load_seqs(self):
        self.seqs = pd.read_csv(self.path_seqs, sep='\t')
        self.seqs.loc[:,'iid'] = self.seqs.loc[:,'iid'].astype('int')
        self.seqs = self.seqs.set_index('iid')
        self.seqs_idx = self.seqs.index
        

    def load_idps(self):
        idps_raw = pd.read_csv(self.path_idps)
        self.idps_map = pd.read_csv(self.path_idps_map) # mapping file 
        idps_filt = idps_raw.dropna(thresh=len(idps_raw) - 1, axis=1) # Drop columns with all NA values on columns
        self.idps_filt = idps_filt.dropna(how= 'any', axis=0) # Drop rows (subjects) with NAs
        self.idps_filt = self.idps_filt.iloc[1:]
        self.idps_filt = self.idps_filt.set_index(self.idps_filt.iloc[:,0].astype('int'))
        self.idps_filt = self.idps_filt.iloc[:,1:]
        self.idps_filt = self.idps_filt[~self.idps_filt.index.duplicated(keep='first')]
        self.idps_idx = self.idps_filt.index

    def match_modalities(self):
        self.matching_ids = np.intersect1d(self.seqs_idx, self.idps_idx)
        self.seqs = self.seqs.loc[self.matching_ids].sort_index()
        self.idps_filt = self.idps_filt.loc[self.matching_ids].sort_index()
        self.iid_tensorID_map = dict(zip(self.seqs.index,np.arange(len(self.seqs.index))))
        self.seqs = self.seqs.reset_index(drop=True)
        self.idps_filt = self.idps_filt.reset_index(drop=True)
    
    def get_iid_tensor_map(self):
        return self.iid_tensorID_map
    
    def map_iid2tensor(self):
        return pd.Index(self.matching_ids).map(self.iid_tensorID_map).values

    def load_data(self):
        self.load_seqs()
        self.load_idps()
        self.match_modalities()

    def set_data_splits(self):
        self.train_idx, self.test_idx, _, _ = train_test_split(
             self.map_iid2tensor(), np.zeros(len(self.matching_ids)), test_size=self.test_size, random_state=self.rnd_st)

        self.train_idx, self.val_idx, _, _  = train_test_split(
            self.train_idx, np.zeros(len(self.train_idx)), test_size = self.val_size, random_state=self.rnd_st) 
    
    def get_data_splits(self):
        return self.train_idx, self.val_idx, self.test_idx

    def create_pairs(self):
        # two possible ways to approach it.
        # If we follow CLIP, we could define pairs of IDPS and SNPs following the GWAS-IDPs paper -> CE loss predicting to what IDP the SNP belongs and vice-versa.
        # If we follow a traditional contrastive framework, we apply a distance loss between the pairs -> same patient being a pair.
        return dict(zip(self.matching_ids,self.matching_ids))
    
    def tokenize_seqs(self):
        # Import the tokenizer and the model
        
        tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-1000g")
        self.seqs_tokenized = tokenizer.batch_encode_plus(self.seqs.values.flatten(), return_tensors="pt")["input_ids"][:,1]

        # seqs_l = self.seqs.values.tolist()
        # self.seqs_t = list()
        # for r in seqs_l:
        #     self.seqs_t.append(tokenizer.batch_encode_plus(r, return_tensors="pt")["input_ids"][:,1])
        # return vstack(self.seqs_t)

    
    def tensorize_data(self):
        # self.seqs = from_numpy(self.seqs.values) # cannot tensorize sequences, that will have to be done in the data loder. Or tokenize first here
        self.idps_filt = from_numpy(self.idps_filt.values.astype('float'))

    def encode_idps(self):
        return 0

    def __init__(self,paths, args):
        # Set paths and args variables
        self.path_seqs, self.path_idps, self.path_idps_map = paths['path_seqs'], paths['path_idps'], paths['path_idp_map']
        self.val_size, self.test_size, self.rnd_st = args['val_size'], args['test_size'], args['rnd_st']

        self.load_data()
        self.set_data_splits()
        # self.encode_idps() # remains to be decided where to perform as this will require to first perfrom the data splits
        self.tokenize_seqs()
        self.pair_dictionary = self.create_pairs()
        self.tensorize_data()
    
    def __len__(self):
        # TODO: Update based on data pair definition
        return len(self.matching_ids)

    def __getitem__(self, index):
        return self.idps_filt[index]#, self.seqs[index], self.tokens_ids
    


# %%
paths = {
    'path_seqs' : '~/ukb-pgx/comical/comical/data/neuroMRI_qced_seqs.csv',
    'path_idps' : '~/ukb-pgx/comical/comical/data/T1_struct_brainMRI_IDPs.csv',
    'path_idp_map' : '~/ukb-pgx/comical/comical/data/T1mri.csv',
}
args = {
    'val_size':0.20,
    'test_size':0.10,
    'rnd_st':0
}

data = dataset(paths, args)
train_idx, val_idx, test_idx = data.get_data_splits()
print(data[test_idx].shape)
assert np.intersect1d(train_idx, val_idx).size == 0 and np.intersect1d(train_idx, test_idx).size == 0 and np.intersect1d(val_idx, test_idx).size == 0