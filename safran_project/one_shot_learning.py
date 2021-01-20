import re
import numpy as np
from PIL import Image
import os
import sys

from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

def make_square(im, size=250, fill_color=(0, 0, 0, 0)):
    new_im = Image.new('RGBA', (size, size), fill_color)
    x,y = new_im.size
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    img0 = new_im.convert('L')
    return img0

def data_augmentation_with_PIL(img):
  image = Image.fromarray(img)
  #horizontal flip
  hoz_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
  #vertical flip
  ver_flip = image.transpose(Image.FLIP_TOP_BOTTOM)

  #rotation 
  rot_90 = image.transpose(Image.ROTATE_90)
  rot_180 = image.transpose(Image.ROTATE_180)
  rot_270 = image.transpose(Image.ROTATE_270)

  #transposition
  transpose = image.transpose(Image.TRANSPOSE)
  transverse = image.transpose(Image.TRANSVERSE)

  return([make_square(hoz_flip), make_square(ver_flip), make_square(rot_90), make_square(rot_180), make_square(rot_270), make_square(transpose), make_square(transverse)])

# translation left
def translation_left(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]
  for i in range(HEIGHT-1, 1, -1):
    for j in range(WIDTH):
      if (i >= 20):
        img[j][i] = img[j][i-20]
      else:
        img[j][i] = 0
  return([make_square(Image.fromarray(img))])

# translation right
def translation_right(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]
  for j in range(WIDTH):
    for i in range(HEIGHT):
      if (i < HEIGHT-20):
        img[j][i] = img[j][i+20]
  return([make_square(Image.fromarray(img))])

# translation up 
def translation_up(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]
  for j in range(WIDTH):
    for i in range(HEIGHT):
      if (j < WIDTH - 20 and j >= 0):
        img[j][i] = img[j+20][i]
      else:
        img[j][i] = 0
  return([make_square(Image.fromarray(img))])

# translation down
def translation_down(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]
  for j in range(WIDTH-1, 1, -1):
    for i in range(HEIGHT):
      if (j <= WIDTH and j > 20):
        img[j][i] = img[j-20][i]
      else:
        img[j][i] = 0
  return([make_square(Image.fromarray(img))])
#adding noise
def adding_noise(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]

  noise = np.random.randint(5, size = (WIDTH, HEIGHT, 1), dtype = 'uint8')

  for i in range(WIDTH):
      for j in range(HEIGHT):
          if (img[i][j] != 255):
              img[i][j] += noise[i][j]
  
  return([make_square(Image.fromarray(img))])

#adding noise
def adding_dead_pixels(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]

  noise = np.random.randint(1000, size = (WIDTH, HEIGHT))

  for i in range(WIDTH):
      for j in range(HEIGHT):
          if (noise[i][j] == 3):
              img[i][j] =0
          elif (noise[i][j] == 122):
              img[i][j] =255
  
  return([make_square(Image.fromarray(img))]) 
#up oscurity
def adding_more_obscurity(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]
  for i in range(WIDTH):
      for j in range(HEIGHT):
          if (int(img[i][j] * 1.1) >255):
              img[i][j] = int(img[i][j] * 1.1)
          else:
              img[i][j] = 255
  
  return([make_square(Image.fromarray(img))])

#up light
def adding_more_light(img):
  HEIGHT, WIDTH = img.shape[1], img.shape[0]

  for i in range(WIDTH):
      for j in range(HEIGHT):
          img[i][j] = int(img[i][j] * 0.9)

  return([make_square(Image.fromarray(img))])



list_functions_augmentation = [data_augmentation_with_PIL, translation_left, translation_right, translation_up, translation_down, adding_noise,\
                               adding_dead_pixels, adding_more_obscurity, adding_more_light]

def transform_data(path, path_data, size):
  try: 
      os.mkdir(path_data + '/data')
  except:
      pass

  for filename in os.listdir(path):
      print("___________________")
      print(filename)
      if filename != '.DS_Store':
          int_ = 0
          try: 
              os.mkdir(path_data + '/data/'+str(filename[-1]))
          except:
              pass
          for imgname in os.listdir(path+'/'+filename):
              print("- - - - - - - - - - - - -")
              print(imgname)
              if imgname != '.DS_Store':
                  img = Image.open(path+"/"+filename +"/"+imgname)
                  img.thumbnail((size,size))
                  print(img.size)
                  print("+++++++++")
                  img0 = img.convert('L') # conversion en niveau de gris


                  images = [make_square(img0)]

                  img_ = np.array(img0)

                  for function in list_functions_augmentation:
                    images += function(img_)

                  for imgs in images:
                    print(imgs.size)
                    int_+=1
                    imgs.save(path_data + '/data/'+filename[-1]+'/'+str(int_)+'.pgm')

def read_image(filename, byteorder='>'):
    
    #first we read the image, as a raw file to the buffer
    with open(filename, 'rb') as f:
        buffer = f.read()
    
    #using regex, we extract the header, width, height and maxval of the image
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    
    #then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



def get_data(path_data,number_of_places,number_of_pic_per_place, size, total_sample_size):
    #read the image
    image = read_image(path_data+ "/data/"+ str(1) + '/' + str(1) + '.pgm', 'rw+')
    #reduce the size
    image = image[::size, ::size]
    #get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0
    
    #initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])
    
    number_of_pictures = len(os.listdir(path_data+"/"+str(1)))-1
    
    for i in range(number_of_places):
        for j in range(int(total_sample_size/number_of_places)):
            ind1 = 0
            ind2 = 0
            
            #read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(number_of_pictures)
                ind2 = np.random.randint(number_of_pictures)
            
            # read the two images
            img1 = read_image(path_data + str(i+1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image(path_data + str(i+1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')
            print(img1.shape)
            print(img2.shape)
            #reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            #store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2
            
            #as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])
    
    for i in range(int(total_sample_size/number_of_pic_per_place)):
        for j in range(number_of_pic_per_place):
            
            #read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(number_of_places)
                ind2 = np.random.randint(number_of_places)
                if ind1 != ind2:
                    break
                    
            img1 = read_image(path_data + str(ind1+1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image(path_data + str(ind2+1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            #as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1
            
    #now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

def build_base_network(input_shape):
    
    seq = Sequential()
    
    nb_filter = [6, 12]
    kernel_size = 3
    
    
    #convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape,
                          padding='same', data_format="channels_first"))
    seq.add(Activation('relu'))
    # seq.add(Conv2D(32, (3, 3), input_shape=input_shape)) 
    seq.add(MaxPooling2D(pool_size=(2, 2))) 
    seq.add(Dropout(.25))
    
    #convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size, padding='same', data_format="channels_first"))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first")) 
    seq.add(Dropout(.25))

    #flatten 
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    return labels[predictions.ravel() < 0.5].mean()

def main(path, path_data):
  # path = './cave/'
  # path_data = "./data/"
  size_wanted = 250

  size = 2
  total_sample_size = 10000
  number_of_places = 7
  number_of_pic_per_place = 128

  transform_data(path, path_data, size_wanted)

  X, Y = get_data(path_data, total_sample_size,number_of_places,number_of_pic_per_place, size)
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

  input_dim = x_train.shape[2:]
  img_a = Input(shape=input_dim)
  img_b = Input(shape=input_dim)

  base_network = build_base_network(input_dim)
  feat_vecs_a = base_network(img_a)
  feat_vecs_b = base_network(img_b)

  distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

  epochs = 13
  rms = RMSprop()

  model = Model(inputs=[img_a, img_b], outputs=distance)
  model.compile(loss=contrastive_loss, optimizer=rms)

  img_1 = x_train[:, 0]
  img2 = x_train[:, 1]

  model.fit([img_1, img2], y_train, validation_split=.25,
            batch_size=128, verbose=2, epochs=epochs)

  pred = model.predict([x_test[:, 0], x_test[:, 1]])
  print(model.summary())
  print(compute_accuracy(pred, y_test))
  

if __name__ == "__main__":
    path__ =  os.path.abspath(sys.argv[1])
    path_data = os.path.abspath(sys.argv[2])
    main(path__, path_data)