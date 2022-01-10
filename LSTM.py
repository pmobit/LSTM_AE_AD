#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:06:17 2021

@author: pooyan
"""

# Importing Libraries

import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import librosa
#from keras.layers import Dropout
#import Audio_data
#import Labels
#import librosa
# sklearn Preprocessing
#from sklearn.model_selection import train_test_split
#import Evaluation
import process
import LSTM_train
from Evaluation import *
import dataset
#from

# load datasets and data processing

#dirname = '/Users/pooyan/Desktop/cmms'
#for filename in glob.glob(os.path.join(dirname, '*.ogg')):

#def get_file_paths(dirname):
 #   file_paths = []  
  #  for root, directories, files in os.walk(dirname):
   #     for filename in files:
    #        filepath = os.path.join(root, filename)
     #       file_paths.append(filepath)  
    #return file_paths  

#class AudioDataset(Dataset):
 #   def __init__(self, filepath):
  #      files = os.listdir(filepath)
   #     self.labels = []
    #    self.file_names = []
     #   for i in files:
      #      x = i.split('_')
       #     self.file_names.append(i)
        #    self.labels.append((int)(x[1]))
        #self.file_path = filepath
        
    #def __getitem__(self, index):
     #   path = self.file_path + str(self.file_names[index]) #+ '.wav'
      #  sound = torch.load(path)
       # soundData = sound
        #return soundData, self.labels[index]

    #def __len__(self):
     #   return len(self.file_names)
  X_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
  X_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
  
  #(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
 # Convert features and corresponding classification labels into numpy arrays
#X = np.array(featuresdf.feature.tolist())
#y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
#le = LabelEncoder()
#yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
#from sklearn.model_selection import train_test_split 

#x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
   
     #set winsow size
     
   #window = 50  
     
#def windows(data, window_size):
 #   start = 0
  #  while start < len(data):
   #    yield start, start + window_size
    # start += (window_size / 2)
     
    
   # import split_folders
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
#split_folders.ratio('./audio_data/', output="./data", seed=1337, ratio=(.8, .2))
    
# Create an audioFeatureExtractor object 
#%to extract the centroid and slope of the mel spectrum over time.
    
#mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)



#treat the extracted features as sequences and use a
#sequenceInputLayer as the first layer of your deep learning model=
    # Reshape data to format for LSTM



#Extract the validation features.


# Initialize LSTM model 5layers

# Hyper Parameters

# hidden_size = 50
# num_layers = 5
# num_classes = 2
# num_hopspersequence=size(featuresTrain)

# data_Size = normal+anomalies
#num_epochs = 3

#sequence,lstm,fully connected, softmax, classification
# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(50, input_shape=(50,1)))
model.compile(optimizer='adam', loss='mse')
model.add(Dense(units=1))
model.add(Dense(number_of_classes,activation='softmax'))
model.add(Dropout())

# m=model
#m = Sequential()
#m.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1],1)))
#m.add(Dropout(0.2))
#m.add(LSTM(units=50))
#m.add(Dropout(0.2))
#m.add(Dense(units=1))
#m.compile(optimizer = 'adam', loss = 'mean_squared_error') #loss=mse

#model.compile(optimizer='adam', loss='mse')


#training options and train the network,


#test the network


#prediction, clasification
#model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)

#model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
X_train_pred = model.predict(X_train, verbose=0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)

#confusion matrix result
model = LogisticRegression()
model = model.fit(matrix, labels)
pred = model.predict(test_matrix)
cm=metrics.confusion_matrix(test_labels,pred)
print(cm)
plt.imshow(cm, cmap='binary')