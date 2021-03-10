def grid_search(batch_size_list, epochs_list, lr_list, path_input, 
                path_csv_output, size_split, size_picture, ratio=1):

      ##### INITIALIZATION ######   
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #device = "cpu"
      print('device :',device)
      dict_results = {}
      
      data_transform = transforms.Compose([transforms.ToTensor()])
      # dataloaders = get_dataloaders(path_input,path_csv_output, batch_size,ratio, data_transform, size_split, False)
      datasets = train_split_dataset(path_input,path_csv_output,ratio, data_transform,size_split, False)

      print('##############################')
      for batch_size in batch_size_list:

        train_loader = torch.utils.data.DataLoader(datasets[0], batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(datasets[1], batch_size=batch_size, shuffle=True, num_workers=0)

        for epochs in epochs_list:
          for lr in lr_list:
            #torch.cuda.empty_cache()

            print('Working on model = ', ' batch size: '+str(batch_size)+' epochs: '+str(epochs) +' lr: '+str(lr))
            network = SiameseNetwork(size_picture).to(device)
            network = nn.DataParallel(network)

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
            print('______________')

      return(dict_results)



class CustomDataset(Dataset):
    """Segmentation & Classification dataset."""

    def _init_(self, folder_inputs,path_csv,list_indexes,ratio=0, transform=None, train=True):
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

      for filename in filenames:
          class_i = df_output[df_output['path'] == filename]['class'].values[0]

          # true pairs
          filenames_int_i = df_output[df_output['class'] == class_i]['path'].values
          add_pairs = [[filename, filename_2] for filename_2 in filenames_int_i if filename!=filename_2]
          #print('__________')
          #print('new pairs: ', len(add_pairs))

          if len(add_pairs)!=0:
              pairs += add_pairs
              output += [1 for i in range(len(add_pairs))]

              #print('pairs: ', len(pairs))
              #print('output: ', len(output))

              #false pairs

              filenames_int_not_i = list(df_output[df_output['class'] != class_i].values)
              shuffle(filenames_int_not_i)
              nbr_false_to_keep = min(len(filenames_int_not_i) , int(len(add_pairs) *self.ratio))
              #print(nbr_false_to_keep)
              filenames_int_not_i = filenames_int_not_i[:nbr_false_to_keep]
              pairs += [[filename, filename_2] for filename_2 in filenames_int_not_i]
              output += [1 for i in range(nbr_false_to_keep)]

              #print('pairs: ', len(pairs))
              #print('output: ', len(output))

      return(pairs, output)


    def _len_(self):
        return len(self.landmarks)

    def _getitem_(self, idx):
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
