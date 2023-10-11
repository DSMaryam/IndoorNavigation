print("start import")
import cyvlfeat
from skimage.io import imread
# from skimage.transform import resize, rotate
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import warnings
# #from cyvlfeat.plot import plotframes
# from scipy.io import loadmat
import numpy as np
from sklearn.cluster import KMeans
import pickle
import datetime
import os, sys
t0= datetime.datetime.now()
print("end import")


# ignore warnings
# warnings.filterwarnings('ignore')

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

if __name__ == "__main__":
    print("start working")
    #folder="/content/gdrive/My Drive/Colab Notebooks/Projet SAFRAN/premiers_tests/dataset_ref_2"
    # folder="C:/Users/Maryam/Documents/000A NON WINDOWS/Cours & TD/3A/Projet SAFRAN/Photos"
    folder=os.path.abspath(sys.argv[1])
    output=os.path.abspath(sys.argv[2])
    nb_cluster=sys.argv[3]
    print(nb_cluster)
    #path vers le dossier contenant toutes les images de référence
    images=[]
    images_loc=[]
    all_words=[]
    docs_lens=[]

    dic={}
    i = 0
    for filename in os.listdir(folder):
            if 'Point' in filename:
                i+=1
                # print(filename, i/len(folder))
                img = imread(os.path.join(folder,filename),pilmode='RGB')
                if img is not None:
                    # print(filename)
                    images.append(img)
                    images_loc.append(str(filename))
                    [frames, document] = cyvlfeat.sift.sift(rgb2gray(img), peak_thresh=0.01,compute_descriptor=True)
                    docs_lens.append(document.shape[0])
                    all_words+=list(document)

    images=np.array(images)
    all_words=np.array(all_words)

    print("##start cluster##")
    n_clusters=int(nb_cluster)
    feature_vocab,labels=compute_vocabulary(all_words,n_clusters)
    hists=compute_hists(docs_lens,labels,n_clusters)
    imdb_tfidf=tf_idf(hists)

    imdb_hists=hists*imdb_tfidf
    imdb_hists = np.sqrt(imdb_hists)
    imdb_hists = imdb_hists/np.linalg.norm(imdb_hists)
    imdb={'images':images,'loc':images_loc,'vocab':feature_vocab,'index':imdb_hists,'idf':imdb_tfidf}
    # pickle.dump( imdb, open( "C:/Users/Maryam/Documents/000A NON WINDOWS/Cours & TD/3A/Projet SAFRAN/Photos/imdb3.p", "wb" ) )
    pickle.dump( imdb, open( output+"/imdb"+nb_cluster+".p", "wb" ) )
    print(datetime.datetime.now() - t0)
