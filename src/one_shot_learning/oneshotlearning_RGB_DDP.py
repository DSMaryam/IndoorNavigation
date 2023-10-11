# -*- coding: utf-8 -*-

## One Shot Learning

### Imports

import re
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import random 
from random import randint, shuffle
import time
import math

#data augmentation imports
from skimage import data
from skimage.transform import resize, rotate
from skimage.util import random_noise
from skimage import exposure
import scipy.ndimage as ndimage

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR


from utils.DDPUtil import set_DDP_device, master_print, DDP_prepare, move_model_to_device, move_to_device, \
    use_multi_GPUs, is_master, get_device

# generate random integer values
"""### Preprocessing of the Data

#### Utilities
"""

def data_augmentation(original_image, new_size, rgb=True):
  original_image = resize(original_image, (new_size[0], new_size[1], new_size[2]),anti_aliasing=True)

  list_augmentation = []

  #original if rgb and grey version if not
  list_augmentation.append(original_image)

  #rotations
  list_augmentation.append(rotate(original_image, 45))
  list_augmentation.append(rotate(original_image, 315))

  #random noise
  list_augmentation.append(random_noise(original_image))

  #horizontal flip
  list_augmentation.append(original_image[:, ::-1,:])

  #blured
  list_augmentation.append(ndimage.uniform_filter(original_image, size=(11)))

  #contrast 
  v_min, v_max = np.percentile(original_image, (0.2, 99.8))
  list_augmentation.append(exposure.rescale_intensity(original_image, in_range=(v_min, v_max)))


  return(list_augmentation)

def make_square(array_, size):
    array_size = array_.shape
    array_output = np.zeros((size[0], size[1], size[2]))
    array_output[0:array_size[0], 0:array_size[1]: , 0:array_size[2]] = array_[0:array_size[0], 0:array_size[1], 0:array_size[2]]
    return(array_output)

"""### Distributed Utilities"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

"""#### Main Preprocessing"""

def transform_data(input_path, path_output,path_output_csv, new_size,  rgb=True, resize=False, ):

  try:
    os.mkdir(path_output)
  except:
    pass
  list_for_outputs = []
  r=0
  j=0
  for folder in os.listdir(input_path):
      f=open(path_csv_results.replace('csv','txt'), 'a')
      f.write("___________________"+"\n")
      f.write(folder)
      f.close()
      
      for imgname in os.listdir(input_path+folder):
          f=open(path_csv_results.replace('csv','txt'), 'a')
          f.write("- - - - - - - - - - - - -"+"\n")
          f.write(imgname)
          f.close()

          #path_picture = "/point_"+str(int_)+"_caption_"+str(j)
          #os.mkdir(path_output+path_picture)
          
          original_image = plt.imread(input_path+folder+'/'+imgname)
          f=open(path_csv_results.replace('csv','txt'), 'a')
          f.write(str(original_image.shape)+"\n")
          f.close()
          images = data_augmentation(original_image, new_size, rgb)
          original_image=True
          f=open(path_csv_results.replace('csv','txt'), 'a')
          f.write(str(images[2].shape)+"\n")
          f.close()
          for image in images:
            image_to_save = make_square(image,  new_size)
            r+=1
            #show_image(image_to_save)

            plt.imsave(path_output+'/'+ str(r)+'.jpg', image_to_save) #, cmap='gray'
            list_for_outputs.append([str(r)+'.jpg', j, original_image])
            if original_image:
              original_image=False

          j+=1

  df = pd.DataFrame(list_for_outputs, columns= ['path', 'class', 'original_image'])
  df.to_csv(path_output_csv+ './output.csv')

"""### Processing DataSet & DataLoader"""


"""#### Custom Dataset Class"""

class CustomDataset(Dataset):
    """Segmentation & Classification dataset."""

    def __init__(self, folder_inputs,path_csv,list_indexes,ratio, transform=None, train=True):
        """
        Args:
            folder_outputs (string): Path to the folder with.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_output = pd.read_csv(path_csv)
        self.folder_inputs = folder_inputs
        self.list_indexes = list_indexes
        self.max_nb_pairs=10000
        if train:
          self.inputs, self.landmarks = self.generate_random_dataset()
        else:
          self.ratio = ratio
          self.inputs, self.landmarks = self.generate_all_pairs_with_ratio()

        self.transform = transform

    def generate_random_dataset(self):
        df_output = self.data_output.iloc[self.list_indexes, :]
        pairs = []
        output = []
        max_nb_pairs_reached=0
        max_n=self.max_nb_pairs//2
        break_2=False
        break_3=False
        for class_ in set(df_output['class'].values):
            df_int = df_output[df_output['class']==class_]
            filenames = list(df_int['path'].values)
            if break_3:
                break
            for i in range(len(filenames)):
                if break_2:
                    break
                for j in range(i+1, len(filenames)):
                    output.append(1)
                    pairs.append([filenames[i], filenames[j]])
                    if max_nb_pairs_reached>max_n:
                        break_2=True
                        break_3=True
                        break
                    max_nb_pairs_reached+=1

        filenames = list(df_output['path'].values)

        for i in range(len(pairs)):
            first_file = filenames[randint(0, len(filenames)-1)]
            class_first_file = int(df_output[df_output['path'] == first_file]['class'].values)
            list_second_file = list(df_output[df_output['class'] != class_first_file]['path'].values)
            output.append(0)
            pairs.append([first_file, list_second_file[randint(0, len(list_second_file)-1)]])
        
        return(pairs, output)

    def generate_all_pairs_with_ratio(self):
      
      pairs = []
      output = []
      
      df_output = self.data_output.iloc[self.list_indexes, :]
      filenames = df_output['path'].values
      max_pairs_per_file=math.floor(self.max_nb_pairs/len(filenames))
      max_real_pairs_per_file=math.floor(max_pairs_per_file/(1+self.ratio))

      for filename in filenames:
          class_i = df_output[df_output['path'] == filename]['class'].values[0]

          # true pairs
          filenames_int_i = df_output[df_output['class'] == class_i]['path'].values
          add_pairs = [[filename, filename_2] for filename_2 in filenames_int_i if filename!=filename_2]
          #print('__________________________')
          #print('new pairs: ', len(add_pairs))

          if len(add_pairs)!=0:
              max_length=min(len(add_pairs),max_real_pairs_per_file)
              shuffle(add_pairs)
              add_pairs=add_pairs[:max_length]
              pairs += add_pairs
              output += [1 for i in range(len(add_pairs))]

              #print('pairs: ', len(pairs))
              #print('output: ', len(output))

              #false pairs

              filenames_int_not_i = list(df_output[df_output['class'] != class_i]['path'].values)
              shuffle(filenames_int_not_i)
              nbr_false_to_keep = min(len(filenames_int_not_i) , int(len(add_pairs) *self.ratio))
              #print(nbr_false_to_keep)
              filenames_int_not_i = filenames_int_not_i[:nbr_false_to_keep]
              pairs += [[filename, filename_2] for filename_2 in filenames_int_not_i]
              output += [0 for i in range(nbr_false_to_keep)]

              #print('pairs: ', len(pairs))
              #print('output: ', len(output))

      return(pairs, output)



    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        

        landmarks = [self.landmarks[idx]]
        image1 = plt.imread(self.folder_inputs+str(self.inputs[idx][0]))
        image2 = plt.imread(self.folder_inputs+str(self.inputs[idx][1]))
        pair = np.zeros([image2.shape[0], image2.shape[1], image2.shape[2]*2])  # 2 is for pairs
        pair[:,:,0] = image1[:,:,0]
        pair[:,:,1] = image1[:,:,1]
        pair[:,:,2] = image1[:,:,2]
        pair[:,:,3] = image2[:,:,0]
        pair[:,:,4] = image2[:,:,1]
        pair[:,:,5] = image2[:,:,2]
        landmarks = np.array([landmarks])

        if self.transform:
            pair = self.transform(pair)
            landmarks = self.transform(landmarks)
            
        return pair, landmarks

"""#### Custom DataLoaders"""

def train_split_dataset(folder_inputs,path_csv,ratio, transform=None,size_split=[], train=True):
  datasets = []
  master_print('###########################')
  master_print(folder_inputs)
  master_print(path_csv)
  master_print(path_csv_results)
  master_print(train)
  master_print(os.path.abspath("."))
  master_print('###########################')
  number_indexes = len(pd.read_csv(path_csv))
  list_indexes = [i for i in range(number_indexes)]
  if train:
      train_dataset = CustomDataset(folder_inputs,path_csv,list_indexes, transform, True)
      datasets = [train_dataset]
  else:
      random.shuffle(list_indexes)

      train_indexes = list_indexes[0: int(size_split[0]*len(list_indexes))]
      test_indexes = list_indexes[int(size_split[0]*len(list_indexes)):]
    #   print(folder_inputs,path_csv,train_indexes,ratio, transform)
      train_dataset = CustomDataset(folder_inputs,path_csv,train_indexes,ratio, transform, True)
      f=open(path_csv_results.replace('csv','txt'), 'a')
      f.write(str(len(train_dataset))+"\n")
      f.close()
      test_dataset = CustomDataset(folder_inputs,path_csv,test_indexes,ratio, transform, False)
      f=open(path_csv_results.replace('csv','txt'), 'a')
      f.write(str(len(test_dataset))+"\n")
      f.close()
      datasets = [train_dataset, test_dataset]

  return datasets

def get_dataloaders(folder_inputs,path_csv, batch_size,ratio,  transform=None,size_split=[], train=True):
  dataloaders = []
  datasets = train_split_dataset(folder_inputs,path_csv,path_csv_results,ratio, transform,size_split, train)
  train_loader = torch.utils.data.DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
  if train:
      return([train_loader])
  else:
      test_loader = torch.utils.data.DataLoader(datasets[1], batch_size=batch_size, shuffle=True, num_workers=0)
      return([train_loader, test_loader])

"""### Model & utilities

#### Siamese Class
"""

class SiameseNetwork(nn.Module):
    def __init__(self, image_size, device):
        super(SiameseNetwork, self).__init__()
        self.image_size = image_size
        self.device = device
        # Layer 1
        self.conv_1 = nn.Conv2d(in_channels=self.image_size[0], out_channels= 8, kernel_size= 3,padding=1)  
        self.activation_1 = nn.ReLU()
        self.max_pooling_1 = nn.MaxPool2d((5, 5))
        self.dropout_1 = nn.Dropout(p=0.25)

        #layer 2
        self.conv_2 = nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3,padding=1)  
        self.activation_2 = nn.ReLU()
        self.max_pooling_2 = nn.MaxPool2d((5, 5))
        self.dropout_2 = nn.Dropout(p=0.25)

        #layer 3
        self.conv_3 = nn.Conv2d(in_channels= 8, out_channels= 8, kernel_size= 3,padding=1)  
        self.activation_3 = nn.ReLU()
        self.max_pooling_3 = nn.MaxPool2d((2, 2))
        self.dropout_3 = nn.Dropout(p=0.25)    

        # layer 4
        self.linear_4 = nn.Linear(200, 100)
        self.activation_4 = nn.ReLU()

        # layer 5: output
        self.linear_5 = nn.Linear(100, 2)

        

    def forward(self,x):
        #print('in', x.shape)
        x = x.float()
        #layer 1
        x = self.conv_1(x)
        #print('1', x.shape)
        x = self.activation_1(x)
        x = self.max_pooling_1(x)
        x = self.dropout_1(x)

        #layer 2
        x = self.conv_2(x)
        #print('2', x.shape)
        x = self.activation_2(x)
        x = self.max_pooling_2(x)
        x = self.dropout_2(x)
        
        #layer 3
        x = self.conv_3(x)
        #print('3', x.shape)
        x = self.activation_3(x)
        x = self.max_pooling_3(x)
        x = self.dropout_3(x)
        
        #layer 4
        #print('4', x.shape)
        x = x.view(x.size()[0],-1)
        x = self.linear_4(x)
        x = self.activation_4(x)
        
        #layer 5: output
        #print('5', x.shape)
        x = self.linear_5(x)
        x = F.softmax(x, dim=1)
        #print('output', x.shape)
        return  x

"""#### Utilities"""

def train(log_interval, model, train_loader, train_data_sampler, optimizer, epoch):
    model.train()

    # DDP Step 4: Manually shuffle to avoid a known bug for DistributedSampler.
    # https://github.com/pytorch/pytorch/issues/31232
    # https://github.com/pytorch/pytorch/issues/31771
    if use_multi_GPUs():
        train_data_sampler.set_epoch(epoch)

    # DDP Step 5: Only record the global loss value and other information in the master GPU.
    if is_master():
        global_cumulative_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = move_to_device(data), move_to_device(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, torch.flatten(target, start_dim=0))
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        # DDP Step 6: Collect loss value and other information from each GPU to the master GPU.
        if use_multi_GPUs():
            # If more information is needed, add to this tensor.
            result = torch.tensor([loss_value], device=get_device())

            dist.barrier()
            # Get the sum of results from all GPUs
            dist.all_reduce(result, op=dist.ReduceOp.SUM)

            # Only master GPU records all the information
            if is_master():
                result = result.tolist()
                global_cumulative_loss += result[0]
        else:
            # use single GPU or CPU
            global_cumulative_loss += loss_value

        if is_master() and batch_idx % log_interval == 0:
            master_print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), global_cumulative_loss))
            # if args.dry_run:
            #     break

def test(model, test_loader):
    model.eval()
    # test_loss = 0
    # correct = 0

    # DDP Step 5: Only record the global loss value and other information in the master GPU.
    if is_master():
        global_cumulative_loss = 0
        global_correct = 0

    with torch.no_grad():

        for data, target in test_loader:
            data, target = move_to_device(data), move_to_device(target)
            output = model(data)
            test_loss_value = F.cross_entropy(output, torch.flatten(target, start_dim=0), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()


            # DDP Step 6: Collect loss value and other information from each GPU to the master GPU.
            if use_multi_GPUs():
                # If more information is needed, add to this tensor.
                result = torch.tensor([test_loss_value, correct], device=get_device())

                dist.barrier()
                # Get the sum of results from all GPUs
                dist.all_reduce(result, op=dist.ReduceOp.SUM)

                # Only master GPU records all the information
                if is_master():
                    result = result.tolist()
                    global_cumulative_loss += result[0]
                    global_correct += result[1]
            else:
                # use single GPU or CPU
                global_cumulative_loss += test_loss_value
                global_correct += correct

    if is_master():
        global_cumulative_loss /= len(test_loader.dataset)
        master_print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            global_cumulative_loss, global_correct, len(test_loader.dataset),
            100. * global_correct / len(test_loader.dataset)))

"""### DDP stuff"""

def run_DDP(rank, world_size, path_input, path_csv_output, size_split, size_picture, path_csv_results, ratio):
    return None

def run_main_DDP(path_input, path_csv_output):
    num_workers =  4
    batch_size = 512
    gamma = 0.7
    lr = 0.01
    epochs = 5
    log_interval = 5
    transform=transforms.Compose([transforms.ToTensor()])
    datasets = train_split_dataset(path_input,path_csv_output,ratio, transform,size_split, False)
    master_print(len(datasets))
    model = SiameseNetwork(size_picture, "device")
    move_model_to_device(model)
    model, train_loader, test_loader, train_data_sampler, test_data_sampler = DDP_prepare(
        train_dataset=datasets[0],
        test_dataset=datasets[1],
        num_data_processes=num_workers,
        global_batch_size=batch_size,
        # In case you have sophisticated data processing function, pass it to collate_fn (i.e., collate_fn of the DataLoader)
        collate_fn=None, model=model)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma) # TODO: check if util

    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        train(log_interval, model, train_loader, train_data_sampler, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()
    end = time.perf_counter()
    master_print("Total Training Time %.2f seconds" % (end - start))

    # TODO save model 

def run_demo(demo_fn, world_size,path_input, path_csv_output, size_split, size_picture, path_csv_results, ratio):
    mp.spawn(demo_fn,
             args=(world_size,path_input, path_csv_output, size_split, size_picture, path_csv_results, ratio),
             nprocs=world_size,
             join=True)

"""### GridSearch OSL"""

def grid_search(batch_size_list, epochs_list, lr_list, path_input, 
                path_csv_output, size_split, size_picture, path_csv_results, ratio):

      ##### INITIALIZATION ######   
      start_time = time.time()
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #device = "cpu"
      f=open(path_csv_results.replace('csv','txt'), 'a')
      f.write('device : '+str(device)+"\n")
      f.close()
      dict_results = {}
      
      data_transform = transforms.Compose([transforms.ToTensor()])
      datasets = train_split_dataset(path_input,path_csv_output,path_csv_results,ratio, data_transform,size_split, False)
      f=open(path_csv_results.replace('csv','txt'), 'a')
      f.write("--- %s seconds for initialisation ---" % (time.time() - start_time)+"\n")
      f.write('##############################'+"\n")
      f.close()
      for batch_size in batch_size_list: # parallelizer imap
        start_time = time.time()
        train_loader = torch.utils.data.DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=1)
        test_loader = torch.utils.data.DataLoader(datasets[1], batch_size=batch_size, shuffle=True, num_workers=1)
        f=open(path_csv_results.replace('csv','txt'), 'a')
        f.write("--- {} seconds for dataloader with batch_size : {}---".format(str(time.time() - start_time), str(batch_size))+"\n")
        f.close()
        for lr in lr_list: # parallelizer
            start_time = time.time()
            network = SiameseNetwork(size_picture, device).to(device)
            # network = nn.DataParallel(network)
            f=open(path_csv_results.replace('csv','txt'), 'a')
            f.write("no DataParallel"+"\n")
            f.close()

            optimizer = optim.SGD(network.parameters(), lr=lr)
            losses = []
            #torch.cuda.empty_cache()
            f=open(path_csv_results.replace('csv','txt'), 'a')
            f.write("--- {} seconds for creation and optimisation with lr : {}---".format(str(time.time() - start_time), str(lr))+"\n")
            f.close()
                
            ##### TRAINING #####
            for epochs in range(max(epochs_list)+1):
                start_time = time.time()    
                f=open(path_csv_results.replace('csv','txt'), 'a')
                f.write('Working on model = '+' batch size: '+str(batch_size)+' epochs: '+str(epochs) +' lr: '+str(lr)+"\n")
                f.close()
                train_loss = train(epochs, network, train_loader, optimizer, device)
                losses +=[train_loss]
                f=open(path_csv_results.replace('csv','txt'), 'a')
                f.write("--- {} seconds for training with epochs : {}---".format(str(time.time() - start_time), str(epochs))+"\n")    
                f.close()

                if epochs in epochs_list:
                    
                    file_snapshot = '/gpfs/workdir/dunoyerg/snapshot/' + 'snapshot'+'_'+str(batch_size)+'_'+str(epochs)+'_'+str(lr)+'_'+'.pt'
                    network_snapshot(network,optimizer,file_snapshot,epochs,train_loss, losses,epochs_list,batch_size,lr)
                    start_time = time.time()  
                    dict_results['model :'+str(batch_size)+' '+str(epochs) +' '+str(lr)]= [test(network, train_loader, optimizer, device, 'train'), test(network, test_loader, optimizer, device, 'test')]
                    f=open(path_csv_results.replace('csv','txt'), 'a')
                    f.write("--- {} seconds for test ---".format(str(time.time() - start_time))+"\n")
                    f.close()
                    f = open(path_csv_results, "a")
                    f.write(str(batch_size)+';'+str(epochs)+';'+ str(lr) +';'+ ';'.join([str(elem) for elem in dict_results['model :'+str(batch_size)+' '+str(epochs) +' '+str(lr)]])+"\n")
                    f.close()
                    torch.save(network.state_dict(), '/gpfs/workdir/dunoyerg/models/' +'/model'+str(batch_size)+str(epochs)+str(lr)+'.pt')
            del network
            f=open(path_csv_results.replace('csv','txt'), 'a')
            f.write('________________________________________'+"\n")
            f.close()
      return(dict_results)

"""### Net work snapshot """

def network_snapshot(network,optimizer,path,epoch,loss,losses,epochs,batch_size,lr):
      torch.save({
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'batch_size' : batch_size,
            'lr' : lr,
            'epochs' : epochs,
            'losses' : losses
            }, path)

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    master_print(n_gpus)
    start_time = time.time()
    ## parameters
    # paths
    # path_input = './data'
    path_input = os.path.abspath(sys.argv[2])+"/"
    # path_csv_output =  './output.csv'
    path_csv_output = os.path.abspath(sys.argv[3])
    path_csv_results = os.path.abspath(sys.argv[4])
    f=open(path_csv_results.replace('csv','txt'), 'a')
    f.write(path_csv_output+","+ path_input+","+path_csv_results+","+path_csv_results+"\n")
    f.close()
    #grid search parameters
    # batch_size_list = [128, 64, 32]
    # epochs_list = [25, 50, 75, 100, 150, 200]
    # lr_list = [0.1, 0.01, 0.001]
    # size_split =  [0.95, 0.05]
    # size_picture = [6,250, 250]
    # ratio=3
    batch_size_list = [256]
    epochs_list = [1]#[i for i in range(11)]
    lr_list = [0.1]
    size_split =  [0.95, 0.05]
    size_picture = [6,250, 250]
    ratio=3

    ## main 
    
    # gpu_id = "0,1,2,3"  # use GPUs 0, 1, 2, 3
    gpu_id = "0"

    set_DDP_device(gpu_id)

    run_main_DDP(path_input, path_csv_output)


    # grid_results = grid_search(batch_size_list, epochs_list, lr_list, path_input, path_csv_output, size_split, size_picture, path_csv_results, ratio)
    # f=open(path_csv_results.replace('csv','txt'), 'a')
    # f.write(str(grid_results)+"\n")
    # f.write("--- %s seconds ---" % (time.time() - start_time)+"\n")
    # f.close()