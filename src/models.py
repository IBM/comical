import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# MLP only as baseline
class mlp_only(nn.Module):
    def __init__(self, config):
        super(mlp_only, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 51

        # MLP
        self.to_model_dims = nn.Linear(64, 1)
        self.h1 = nn.Linear(5603+139+42,self.d_model)
        self.proj_down = nn.Linear(self.d_model,2)

    def mlp(self,x):
            self.dropout = nn.Dropout(0.2)
            x = self.proj_down(self.dropout(F.relu(self.h1(x))))
            return x

    def forward(self, gen,snp_id,idp,idp_id,save_emb,covariates):
        # adapt input to mlp
        idp_features = torch.squeeze(self.to_model_dims(idp.float()),dim=-1)
        res = self.mlp(torch.concat((gen,idp_features,covariates),dim=-1))
        return res

# Added on 2024.01.10 - Use pretrained encoders to perform downstream tasks such as classification and risk score regression - Yet to be debugged
class comical_new_emb_clasf(nn.Module): 
    def __init__(self, config):
        super(comical_new_emb_clasf, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 51 # Number of genetic dictionary (e.g. A>T , C>G, etc.)
        self.dict_size_num_snps = config['num_snps']
        self.dict_size_num_ipds = config['num_idps']
        self.idp_tok_dims = config['idp_tok_dims']
        self.num_classes = config['num_classes']

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.gen_emb = nn.Embedding(self.dict_size_gen, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)
        self.snp_id_emb = nn.Embedding(self.dict_size_num_snps, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)
        self.idp_emb = nn.Linear(self.idp_tok_dims, int(self.d_model/2))
        self.idp_id_emb = nn.Embedding(self.dict_size_num_ipds, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)

        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)
        self.idp_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.idp_transformer_encoder = nn.TransformerEncoder( self.idp_encoder_tf_layer, num_layers=self.num_layers)

        self.ln_final = nn.LayerNorm(self.d_model)
        self.seq_projection = nn.Parameter(torch.zeros(self.d_model, self.d_model)) # switched torch empty for zeros to see if avoids gradient problems
        self.idp_projection = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        

        nn.init.normal_(self.gen_emb.weight, std=0.02)
        nn.init.normal_(self.idp_emb.weight, std=0.02)
        nn.init.normal_(self.snp_id_emb.weight, std=0.01)
        nn.init.normal_(self.idp_id_emb.weight, std=0.01)
        if self.seq_projection is not None:
            nn.init.normal_(self.seq_projection, std=self.d_model ** -0.5)
        if self.idp_projection is not None:
            nn.init.normal_(self.idp_projection, std=self.d_model ** -0.5)
        
        self.gen_mlp = nn.Linear(self.d_model, self.dict_size_num_snps)
        self.idp_mlp = nn.Linear(self.d_model, self.dict_size_num_ipds)

        # MLP
        self.h1 = nn.Linear(self.d_model*2+42,self.d_model)
        self.proj_down = nn.Linear(self.d_model,self.num_classes)
    
    # Get embeddings for plotting - added 4/4/2025
    def get_embeddings(self):
        return self.gen_emb.weight, self.snp_id_emb.weight, self.idp_emb.weight, self.idp_id_emb.weight
    
    # Genetic (modality 1 encoder)
    def encode_gen(self, text, snp_id, save_emb):
        x = torch.concat((self.gen_emb(text),self.snp_id_emb(snp_id)),dim=-1)
        emb = x.detach().cpu().numpy()
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        # x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding 
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.gen_transformer_encoder(x)
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.seq_projection
        assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)

        if save_emb:
            return x, emb
        else:
            return x
    # def encode_gen(self, text, snp_id, save_emb):
    #     x = torch.concat((self.gen_emb(text),self.snp_id_emb(snp_id)),dim=1)
    #     emb = x.detach().cpu().numpy()
    #     x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
    #     x = torch.unsqueeze(x,1)

    #     # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding 
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.gen_transformer_encoder(x)
    #     x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x)

    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), 0] @ self.seq_projection
    #     assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)
        
    #     if save_emb:
    #         return x, emb
    #     else:
    #         return x

    # IDP (modality 2 encoder)
    def encode_idp(self, idp,idp_id,save_emb):
        x = torch.concat((self.idp_emb(idp),self.idp_id_emb(idp_id)),dim=-1)
        emb = x.detach().cpu().numpy()
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        # x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.idp_transformer_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.idp_projection
        assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)
        
        if save_emb:
            return x, emb
        else:
            return x
        
    # def encode_idp(self, idp,idp_id,save_emb):
    #     x = torch.concat((self.idp_emb(idp),self.idp_id_emb(idp_id)),dim=1)
    #     emb = x.detach().cpu().numpy()
    #     x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
    #     x = torch.unsqueeze(x,1)

    #     # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.idp_transformer_encoder(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x)

    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     x = x[torch.arange(x.shape[0]), 0] @ self.idp_projection
    #     assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)
        
    #     if save_emb:
    #         return x, emb
    #     else:
    #         return x
        
    def mlp(self,x):
            self.dropout = nn.Dropout(0.2)
            x = self.proj_down(self.dropout(F.relu(self.h1(x))))
            return x


    def forward(self, gen,snp_id,idp,idp_id,save_emb,covariates):
        # Prepare embeddings
        if save_emb:
            gen_features,gen_emb = self.encode_gen(gen,snp_id,save_emb)
            idp_features,idp_emb = self.encode_idp(idp.float(),idp_id,save_emb)
        else:
            gen_features = self.encode_gen(gen,snp_id,save_emb)
            idp_features = self.encode_idp(idp.float(),idp_id,save_emb)

        # normalized features
        gen_features = gen_features / gen_features.norm(dim=1, keepdim=True)
        idp_features = idp_features / idp_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_gen = logit_scale * gen_features @ idp_features.t()
        # logits_per_idp = logits_per_gen.t()

        # shape = [batch_size, 1]
        res = self.mlp(torch.concat((gen_features,idp_features,covariates),dim=-1))
        # return torch.squeeze(res) # squeeze to remove last dimension (i.e. shape = [batch_size,]) solve problems with MSE loss function
        if save_emb:
            return res, gen_emb,idp_emb
        else:
            return res
    
################################# Implementation following CLIP #################################
class comical_new_emb(nn.Module): 
    def __init__(self, config):
        super(comical_new_emb, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 51 # Number of genetic dictionary (e.g. A>T , C>G, etc.)
        self.dict_size_num_snps = config['num_snps']
        self.dict_size_num_ipds = config['num_idps']
        self.idp_tok_dims = config['idp_tok_dims']

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.gen_emb = nn.Embedding(self.dict_size_gen, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)
        self.snp_id_emb = nn.Embedding(self.dict_size_num_snps, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)
        self.idp_emb = nn.Linear(self.idp_tok_dims, int(self.d_model/2))
        self.idp_id_emb = nn.Embedding(self.dict_size_num_ipds, int(self.d_model/2), max_norm=True, scale_grad_by_freq=True)

        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)
        self.idp_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.idp_transformer_encoder = nn.TransformerEncoder( self.idp_encoder_tf_layer, num_layers=self.num_layers)

        self.ln_final = nn.LayerNorm(self.d_model)
        self.seq_projection = nn.Parameter(torch.zeros(self.d_model, self.d_model)) # switched torch empty for zeros to see if avoids gradient problems
        self.idp_projection = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        

        nn.init.normal_(self.gen_emb.weight, std=0.02)
        nn.init.normal_(self.idp_emb.weight, std=0.02)
        nn.init.normal_(self.snp_id_emb.weight, std=0.01)
        nn.init.normal_(self.idp_id_emb.weight, std=0.01)
        if self.seq_projection is not None:
            nn.init.normal_(self.seq_projection, std=self.d_model ** -0.5)
        if self.idp_projection is not None:
            nn.init.normal_(self.idp_projection, std=self.d_model ** -0.5)
        
        self.gen_mlp = nn.Linear(self.d_model, self.dict_size_num_snps)
        self.idp_mlp = nn.Linear(self.d_model, self.dict_size_num_ipds)
    
    # Genetic (modality 1 encoder)
    def encode_gen(self, text, snp_id, save_emb):
        x = torch.concat((self.gen_emb(text),self.snp_id_emb(snp_id)),dim=1)
        emb = x.detach().cpu().numpy()
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding 
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.gen_transformer_encoder(x)
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.seq_projection
        assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)

        if save_emb:
            return x, emb
        else:
            return x

    # IDP (modality 2 encoder)
    def encode_idp(self, idp,idp_id,save_emb):
        x = torch.concat((self.idp_emb(idp),self.idp_id_emb(idp_id)),dim=1)
        emb = x.detach().cpu().numpy()
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding # positional embedding replaced by snp id-based embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.idp_transformer_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.idp_projection
        assert x.isnan().sum().cpu().item() == 0 # Throw error if all values have become nan, (i.e. meaning model fail to converge)
        
        if save_emb:
            return x, emb
        else:
            return x


    def forward(self, gen,snp_id,idp,idp_id,save_emb):
        # Prepare embeddings
        if save_emb:
            gen_features,gen_emb = self.encode_gen(gen,snp_id,save_emb)
            idp_features,idp_emb = self.encode_idp(idp.float(),idp_id,save_emb)
        else:
            gen_features = self.encode_gen(gen,snp_id,save_emb)
            idp_features = self.encode_idp(idp.float(),idp_id,save_emb)

        # normalized features
        gen_features = gen_features / gen_features.norm(dim=1, keepdim=True)
        idp_features = idp_features / idp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_gen = logit_scale * gen_features @ idp_features.t()
        logits_per_idp = logits_per_gen.t()

        # shape = [batch_size, batch_size]
        if save_emb:
            return logits_per_gen, logits_per_idp, gen_emb,idp_emb
        else:
            return logits_per_gen, logits_per_idp
    



################################# Naive impl #################################
class comical(nn.Module): # example from fairprs
    def __init__(self, config):
        super(comical, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 9000000
        self.dict_size_idp = 5000
        # self.dtype = 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.gen_emb = nn.Embedding(self.dict_size_gen, self.d_model, max_norm=True)
        self.positional_embedding_gen = nn.Parameter(torch.empty(self.dict_size_gen, self.d_model))
        self.idp_emb = nn.Embedding(self.dict_size_idp, self.d_model, max_norm=True)
        self.positional_embedding_idp = nn.Parameter(torch.empty(self.dict_size_idp, self.d_model))

        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)
        self.idp_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.idp_transformer_encoder = nn.TransformerEncoder( self.idp_encoder_tf_layer, num_layers=self.num_layers)

        self.ln_final = nn.LayerNorm(self.d_model)
        self.seq_projection = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.idp_projection = nn.Parameter(torch.empty(self.d_model, self.d_model))
        

        nn.init.normal_(self.gen_emb.weight, std=0.02)
        nn.init.normal_(self.idp_emb.weight, std=0.02)
        nn.init.normal_(self.positional_embedding_gen, std=0.01)
        nn.init.normal_(self.positional_embedding_idp, std=0.01)
        if self.seq_projection is not None:
            nn.init.normal_(self.seq_projection, std=self.d_model ** -0.5)
        if self.idp_projection is not None:
            nn.init.normal_(self.idp_projection, std=self.d_model ** -0.5)
        
        self.gen_mlp = nn.Linear(self.d_model, self.dict_size_gen)
        self.idp_mlp = nn.Linear(self.d_model, self.dict_size_idp)

    def encode_gen(self, text):
        x = self.gen_emb(text)  # [batch_size, n_ctx, d_model]
        x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.gen_transformer_encoder(x)
        x = torch.nan_to_num(x,0) # TODO: this is a patch fix to get rid of nans
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.seq_projection
        assert x.isnan().sum().cpu().item() == 0 

        return x

    def encode_idp(self, idp):
        x = self.idp_emb(idp)  # [batch_size, n_ctx, d_model]
        x = torch.unsqueeze(x,1)

        # x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.idp_transformer_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), 0] @ self.idp_projection

        return x


    def forward(self, gen,idp):
        # Prepare embeddings
        gen_features = self.encode_gen(gen)
        idp_features = self.encode_idp(idp)

        # normalized features
        gen_features = gen_features / gen_features.norm(dim=1, keepdim=True)
        idp_features = idp_features / idp_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_gen = logit_scale * gen_features @ idp_features.t()
        logits_per_idp = logits_per_gen.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_gen, logits_per_idp

    



################################# Naive impl #################################
class comical_old(nn.Module): # example from fairprs
    def __init__(self, config):
        super(comical_old, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 200
        self.dict_size_idp = 278*2

        self.gen_emb = nn.Embedding(self.dict_size_gen, self.d_model, max_norm=True)
        self.idp_emb = nn.Embedding(self.dict_size_idp, self.d_model, max_norm=True)

        self.gen_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.gen_transformer_encoder = nn.TransformerEncoder( self.gen_encoder_tf_layer, num_layers=self.num_layers)
        self.idp_encoder_tf_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward, dropout=self.dropout, layer_norm_eps= self.layer_norm_eps, activation=self.activation)
        self.idp_transformer_encoder = nn.TransformerEncoder( self.idp_encoder_tf_layer, num_layers=self.num_layers)

        self.gen_mlp = nn.Linear(self.d_model, self.dict_size_gen)
        self.idp_mlp = nn.Linear(self.d_model, self.dict_size_idp)


    def forward(self, gen,idp):
        # Prepare embeddings
        gen = self.gen_emb(gen)
        idp = self.idp_emb(idp)
        # Genetic encoder
        gen = self.gen_transformer_encoder(torch.swapaxes(gen,0,1))
        # IDP encoder
        idp = self.idp_transformer_encoder(torch.swapaxes(idp,0,1))
        # Cross attention

        # Genetic encoder
        gen_logits = self.gen_mlp(torch.swapaxes(gen,0,1)[0])
        # IDP encoder
        idp_logits = self.gen_mlp(torch.swapaxes(idp,0,1)[0])

        return gen_logits, idp_logits
    
############################# Transformer + MLP only for baseline #################################
# Added on 2025.01.23
# MLP only as baseline
class transformer_mlp(nn.Module):
    def __init__(self, config):
        super(mlp_only, self).__init__()
        self.units = config['units']
        self.num_layers = config['num_layers']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']
        self.layer_norm_eps = config['layer_norm_eps']
        self.activation = config['activation']
        self.dict_size_gen = 51

        # MLP
        self.to_model_dims = nn.Linear(64, 1)
        self.h1 = nn.Linear(5603+139+42,self.d_model)
        self.proj_down = nn.Linear(self.d_model,2)

    def mlp(self,x):
            self.dropout = nn.Dropout(0.2)
            x = self.proj_down(self.dropout(F.relu(self.h1(x))))
            return x

    def forward(self, gen,snp_id,idp,idp_id,save_emb,covariates):
        # adapt input to mlp
        idp_features = torch.squeeze(self.to_model_dims(idp.float()),dim=-1)
        res = self.mlp(torch.concat((gen,idp_features,covariates),dim=-1))
        return res


# %% Debugging
# batch_size = 64
# n_gen = 5063
# n_idp = 278

# gen = torch.randint(0,64,(batch_size,))
# idp = torch.randint(0,278*2,(batch_size,))
# config ={
#             ## Training hyperparms
#             "lr": 0.00001,
#             "batch_size" : 64,
#             ## Model hyperparams
#             'units': 16,
#             'num_layers' : 2,
#             'd_model' : 256, 
#             'nhead': 2,
#             'dim_feedforward' : 256,
#             'dropout' : 0.1,
#             'layer_norm_eps' : 0.000001,
#             'activation' : 'relu',
#         }
# model = comical(config)

# gen_logits, idp_logits = model(gen,idp)
# print(gen_logits.shape, idp_logits.shape)
