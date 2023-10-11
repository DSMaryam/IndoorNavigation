## One Shot Learning

### Imports

import re
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import math
import random 
from random import randint, shuffle
import time

#data augmentation imports
from skimage import data
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage import exposure
import scipy.ndimage as ndimage

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray



photo_apprentissage_path=os.path.abspath(os.getcwd())

print("imports termines")

# generate random integer values
"""### Preprocessing of the Data

#### Utilities
"""

def data_augmentation(original_image, new_size, rgb=True):
  original_image = resize(original_image, (new_size[0], new_size[1]),anti_aliasing=True)

  list_augmentation = []

  #original if rgb and grey version if not
  list_augmentation.append(original_image)

  #rotations
  list_augmentation.append(rotate(original_image, 45))
  list_augmentation.append(rotate(original_image, 315))

  #random noise
  list_augmentation.append(random_noise(original_image))

  #horizontal flip
  list_augmentation.append(original_image[:, ::-1])

  #blured
  list_augmentation.append(ndimage.uniform_filter(original_image, size=(11)))

  #contrast 
  v_min, v_max = np.percentile(original_image, (0.2, 99.8))
  list_augmentation.append(exposure.rescale_intensity(original_image, in_range=(v_min, v_max)))


  return(list_augmentation)

def transform_data(input_path, path_output,path_output_csv, new_size):

  try:
    os.mkdir(path_output)
  except:
    pass
  list_for_outputs = []
  r=0
  j=0
  for folder in os.listdir(input_path):
      for imgname in os.listdir(input_path+folder):
          #path_picture = "/point_"+str(int_)+"_caption_"+str(j)
          #os.mkdir(path_output+path_picture)
          
          original_image = plt.imread(input_path+folder+'/'+imgname)
          original_image = rgb2gray(original_image)
          images = data_augmentation(original_image, new_size)

          original_image=True
          for image in images:
            image_to_save = image
            r+=1
            #show_image(image_to_save)

            plt.imsave(path_output+'/'+ str(r)+'.jpg', image_to_save, cmap='gray') #, cmap='gray'
            list_for_outputs.append([str(r)+'.jpg', j, original_image,folder])
            if original_image:
              original_image=False

          j+=1

  df = pd.DataFrame(list_for_outputs, columns= ['path', 'class', 'original_image','position'])
  #df.to_csv(path_output_csv+ '/output_bouygues.csv')
  return df




df = transform_data(photo_apprentissage_path+"/photos-bouygues/", photo_apprentissage_path+'/apprentissage-bouygues-gray/','', new_size=[250,250])
df.to_csv(photo_apprentissage_path+'/output_bouygues-gray.csv')

