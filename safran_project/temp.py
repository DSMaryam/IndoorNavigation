def grid_search(batch_size_list, epochs_list, lr_list, path_input, 
                path_csv_output, size_split, size_picture, ratio=0):

      ##### INITIALIZATION ######   
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #device = "cpu"
      print('device :',device)
      dict_results = {}

      print('##############################')
      for batch_size in batch_size_list:

        data_transform = transforms.Compose([transforms.ToTensor()])
        dataloaders = get_dataloaders(path_input,path_csv_output, batch_size,ratio, data_transform, size_split, False)

        train_loader, test_loader = dataloaders[0], dataloaders[1]

        for epochs in epochs_list:
          for lr in lr_list:
            #torch.cuda.empty_cache()

            print('Working on model = ', ' batch size: '+str(batch_size)+' epochs: '+str(epochs) +' lr: '+str(lr))
            network = SiameseNetwork(size_picture, device).to(device)
            network = nn.DataParallel(network) # try use multiple GPU

            optimizer = optim.SGD(network.parameters(), lr=lr)
            losses = []
            ##### TRAINING #####
            for epoch in range(epochs):
                train_loss = train(epoch, network, train_loader, optimizer, device)
                losses +=[train_loss]
                if epoch%50==0:
                    file_snapshot = '/gpfs/workdir/dunoyerg/snapshot/' + 'snapshot'+'_'+str(batch_size)+'_'+str(epoch)+'_'+str(lr)+'_'+'.pt'
                    network_snapshot(network,optimizer,file_snapshot,epoch,train_loss, losses,epochs_list,batch_size,lr)
                    
            dict_results['model :'+str(batch_size)+' '+str(epochs) +' '+str(lr)]= [test(network, train_loader, optimizer, device, 'train'), test(network, test_loader, optimizer, device, 'test')]
            f = open("results.csv", "a")
            f.write(str(batch_size)+';'+str(epochs)+';'+ str(lr) + ';' + ''.join([str(elem) for elem in dict_results['model :'+str(batch_size)+' '+str(epochs) +' '+str(lr)]])+"\n")
            f.close()
            torch.save(network.state_dict(), '/gpfs/workdir/dunoyerg/models/' +'/model'+str(batch_size)+str(epochs)+str(lr)+'.pt')
            del network
            print('________________________________________')

      return(dict_results)


class CustomDataset(Dataset):
    """Segmentation & Classification dataset."""

    def __init__(self, folder_inputs,path_csv,list_indexes,ratio=0, transform=None, train=True):
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
        for class_ in set(df_output['class'].values):
            df_int = df_output[df_output['class']==class_]
            filenames = list(df_int['path'].values)
            for i in range(len(filenames)):
              for j in range(i+1, len(filenames)):
                  output.append(1)
                  pairs.append([filenames[i], filenames[j]])

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
      
      for i in range(len(filenames)):
        for j in range(i+1, len(filenames)):
            if len(set(df_output[(df_output['path'] == filenames[i]) | (df_output['path'] == filenames[j])]['class'].values))==1:
              output.append(1)
            else: 
              output.append(0)
            pairs.append([filenames[i], filenames[j]])
      
      if self.ratio !=0:
              indices_false = [i for i in range(len(output)) if output[i] == 0]
              indices_true = [i for i in range(len(output)) if output[i] == 1]
              shuffle(indices_false)
              indices_to_keep = indices_false[0:int(len(indices_true)*self.ratio)]+indices_true

              return([pairs[i] for i in range(len(pairs)) if i in indices_to_keep],
              [output[i] for i in range(len(output)) if i in indices_to_keep])

      return(pairs, output)


    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        

        landmarks = [self.landmarks[idx]]
        image1 = plt.imread(self.folder_inputs+self.inputs[idx][0])
        image2 = plt.imread(self.folder_inputs+self.inputs[idx][1])
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