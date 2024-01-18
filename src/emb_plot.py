import torch
import numpy as np
from sklearn.manifold import TSNE
import pickle
import seaborn as sns
import umap
import matplotlib.pyplot as plt
import os



def emb_plot(save_path, embs_path = None, embs = None, tsne_umap = 'tsne', num_points_plot = 500000):
    print('This process can take a while, especially with a large dataset')
    print('Loading data')
    if embs == None:
        with open(embs_path, "rb") as outfile:
            embs = pickle.load(outfile)

    print('Preparing data for plotting')
    gen_emb = np.asarray(embs['gen_embs'])
    idp_emb = np.asarray(embs['idp_embs'])
    del embs

    print(f'Computing {tsne_umap} for genetic embeddings')
    if int(num_points_plot) > 30000000:
        num_points_plot = 500000
    if tsne_umap == 'tsne':
        X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(gen_emb[:num_points_plot])
    else:
        reducer = umap.UMAP()
        X_embedded = reducer.fit_transform(gen_emb[:int(num_points_plot)])

    print('Plotting Genetic embeddings')
    sns.scatterplot(x = X_embedded[:,0], y =X_embedded[:,1])
    plt.savefig(os.path.join(save_path,'gen_emb_tsne.png'))
    del X_embedded
    plt.clf()

    print(f'Computing {tsne_umap} for IDP embeddings')
    if tsne_umap == 'tsne':
        X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(gen_emb[:int(num_points_plot)])
    else:
        reducer = umap.UMAP()
        X_embedded = reducer.fit_transform(gen_emb[:num_points_plot])

    print('Plotting IDP embeddings')
    sns.scatterplot(x = X_embedded[:,0], y =X_embedded[:,1])
    plt.savefig(os.path.join(save_path,'idp_emb_tsne.png'))


# Debugging

# save_path = '../results/'
# embs_path =  '../results/top_1_per_4e_7500bz/embs.pickle'
# emb_plot(save_path,embs_path)