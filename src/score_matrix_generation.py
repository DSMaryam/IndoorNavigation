import cyvlfeat
import numpy as np
from skimage.io import imread
from skimage.transform import resize, rotate
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from cyvlfeat.plot import plotframes
from scipy.io import loadmat
import numpy as np
import sys
import pickle
import os
import pandas as pd
from sklearn.neighbors import KDTree

# change some default matplotlib parameters
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.dpi'] = 120

# ignore warnings
warnings.filterwarnings('ignore')

def rgb2gray(rgb):
    return np.float32(np.dot(rgb[...,:3], [0.299, 0.587, 0.114])/255.0)


def scores_dic(painting, distance_n=2):

  global feature_vocab
  global imdb_hists
  global imdb_tfidf
  global images
  global images_loc

  global num_words
  global num_paintings

  scores_dic={}

  [frames, descrs] = cyvlfeat.sift.sift(rgb2gray(painting), peak_thresh=0.01,compute_descriptor=True) 

  tree = KDTree(feature_vocab)   
  distNN, indNN = tree.query(descrs, k=1) 

  query_hist=np.zeros((num_words))
  for i in range(indNN.shape[0]):
    query_hist[indNN[i]]=1

  # process histogram
  query_hist = query_hist*imdb_tfidf
  query_hist = np.sqrt(query_hist)
  query_hist = query_hist/np.linalg.norm(query_hist)


  for i in range(num_paintings):
    histi=imdb_hists[i]
    scores_dic[images_loc[i]]=1/(np.sum((np.abs(histi-query_hist))**distance_n))**(1/distance_n)
  
  return scores_dic


if __name__=="__main__":
    print("start matrix generation")
    # imdb = pickle.load( open( "/content/gdrive/My Drive/Colab Notebooks/Projet SAFRAN/premiers_tests/imdb0.p", "rb" ) )
    nb_cluster = sys.argv[4]
    imdb = pickle.load(open(os.path.abspath(sys.argv[1])+"/imdb"+nb_cluster+".p", "rb"))
    folder_test = os.path.abspath(sys.argv[2])
    output_dir = os.path.abspath(sys.argv[3])
    distance_n = int(sys.argv[5])

    feature_vocab=imdb['vocab']
    imdb_hists=imdb['index']
    imdb_tfidf=imdb['idf']
    images=imdb['images']
    images_loc=imdb['loc']

    num_words = feature_vocab.shape[0]
    num_paintings = imdb_hists.T.shape[1]
    
    matrix_dic={}

    for filename in os.listdir(folder_test):
        if 'Test' in filename:
            img = imread(os.path.join(folder_test,filename),pilmode='RGB')
            if img is not None:
                matrix_dic[filename]=scores_dic(img, distance_n)     


    # pickle.dump( matrix_dic, open( "/content/gdrive/My Drive/Colab Notebooks/Projet SAFRAN/premiers_tests/test_matrix.p", "wb" ) )
    pickle.dump( matrix_dic, open( output_dir+"test_matrix"+nb_cluster+".p", "wb" ) )
    df= pd.DataFrame(matrix_dic)
    # df.to_csv('/content/gdrive/My Drive/Colab Notebooks/Projet SAFRAN/premiers_tests/score_matrix.csv', index = True) 
    df.to_csv(output_dir+"/score_matrix-clus_"+nb_cluster+"-dist_"+str(distance_n)+".csv", index = True)
