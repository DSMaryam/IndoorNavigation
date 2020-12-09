from plot import plotframes
import cyvlfeat
import numpy as np
from skimage.io import imread
from skimage.transform import resize, rotate
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat
import numpy as np
import os
from sklearn.cluster import KMeans
import pickle as pkl

if __name__ == "__main__":
    # ignore warnings
    warnings.filterwarnings('ignore')

    def rgb2gray(rgb):
        return np.float32(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255.0)

    def compute_hists(corpus_lens,labels,num_words):
        num_docs=len(corpus_lens)
        hists=np.zeros((num_docs,num_words))
        offset=0
        for j in range(num_docs):
            len_doc=corpus_lens[j]
        for i in range(offset,offset+len_doc):
            hists[j,labels[i]]=1
        offset+=len_doc
        return hists

    def tf_idf(hists):
        num_docs=hists.shape[0]
        dfs=np.sum(hists,axis=0)
        dfs=np.log(num_docs/dfs)
        return dfs

    def compute_vocabulary(words,n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(words)
        return kmeans.cluster_centers_,kmeans.labels_

    folder="./dataset_ref_2"
    #path vers le dossier contenant toutes les images de référence
    images=[]
    all_words=[]
    docs_lens=[]
    print('start_preprocessing')
    for filename in os.listdir(folder):
            img = imread(os.path.join(folder,filename),pilmode='RGB')
            if img is not None:
                images.append(img)
                [frames, document] = cyvlfeat.sift.sift(rgb2gray(img), peak_thresh=0.01, compute_descriptor=True)
                docs_lens.append(document.shape[0])
                all_words+=list(document)
    print('done')
    images=np.array(images)
    all_words=np.array(all_words)


    n_clusters=1000
    feature_vocab,labels=compute_vocabulary(all_words,n_clusters)
    hists=compute_hists(docs_lens,labels,n_clusters)
    imdb_tfidf=tf_idf(hists)

    imdb_hists=hists*imdb_tfidf
    imdb_hists = np.sqrt(imdb_hists)
    imdb_hists = imdb_hists/np.linalg.norm(imdb_hists)

    imdb={'images':images,'vocab':feature_vocab,'index':imdb_hists,'idf':imdb_tfidf}

    pkl.dump(imdb, open('./data_clustered', 'wb'))