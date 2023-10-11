from __future__ import print_function

#overall imports
import os
import numpy as np
import pandas as pd
import skimage as im
import matplotlib.pyplot as plt
from scipy import ndimage
import random
import matplotlib.image as saving
import PIL.Image as Image

#data augmentation imports
from skimage import data
from skimage.transform import rescale
from skimage.util import random_noise
from skimage import exposure

import logging

#Model Preparation
from sklearn.model_selection import train_test_split

#other
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())