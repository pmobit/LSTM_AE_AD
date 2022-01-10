#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:23:00 2021

@author: pooyan
"""
### import required libraries and pkgs ###
import glob
import os
import librosa
import numpy as np
#from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt





### Define feature extraction ### defualt=1024 in MATLAB
#MytimeWindow=1024
MySampleRate=44100
#y=audio time series
#sr=sample rate
#fn_nfft=fft length 
#win_length=window size
#hop_length= overlap length




#get all normal and anomalies files from directory

parent_dir = '/Users/pooyan/Desktop/cmms/class/'
sub_dir = 'normal/'
file_ext='*.ogg'
#all_Normalfiles = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))
all_files = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))



#test for a single file
FilePath = all_files[0]

#load file

y,sr = librosa.load(FilePath, MySampleRate)


start_i = 0
end_i = len(y) # to read whole file

signal = y[start_i:end_i]
centroid = librosa.feature.spectral_centroid(y=signal,sr=sr)
slope = librosa.feature.melspectrogram(y=signal,sr=sr, n_mels=1)

my_features = np.concatenate((centroid,slope),axis=0)

#check the dimentions
print(y.shape)
print(signal.shape)
print(centroid.shape)
print(slope.shape)
print(my_features.shape)


def Myextract_features(filePath, sampleRate=44100):
    
    signal,sr = librosa.load(filePath, sampleRate)

    centroid = librosa.feature.spectral_centroid(y=signal,sr=sr)
    slope = librosa.feature.melspectrogram(y=signal,sr=sr, n_mels=1)

    features = np.concatenate((centroid,slope),axis=0)

    return features


my_features = Myextract_features(all_files[1])

#read all noraml files
parent_dir = '/Users/pooyan/Desktop/cmms/class/'
sub_dir = 'normal/'
file_ext='*.ogg'
all_Normalfiles = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[0:350]

all_noraml_features = [] #define an empty array
all_noraml_labels = []
for i in range(0,len(all_Normalfiles)):
    my_features = Myextract_features(all_Normalfiles[i])
    all_noraml_features.append(my_features)
    all_noraml_labels.append("normal")

#use reshape to conver outputs into arrays
dim_1 = len(all_Normalfiles)
dim_2 = my_features.shape[0] #number of features
dim_3 = my_features.shape[1]
all_noraml_features = np.asarray(all_noraml_features).reshape(dim_1, dim_2, dim_3)
all_noraml_labels = np.asarray(all_noraml_labels).reshape(dim_1)

#check the dimentions
print(all_noraml_features.shape)
print(all_noraml_labels.shape)

#read all anoamly files
parent_dir = '/Users/pooyan/Desktop/cmms/class/'
sub_dir = 'anomaly/'
file_ext='*.ogg'
all_Anomalyfiles = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))[0:350]

all_anomaly_features = [] #define an empty array
all_anomaly_labels = []
for i in range(0,len(all_Anomalyfiles)):
    my_features = Myextract_features(all_Anomalyfiles[i])
    all_anomaly_features.append(my_features)
    all_anomaly_labels.append("anomaly")

#use reshape to conver outputs into arrays
dim_1 = len(all_Anomalyfiles)
dim_2 = my_features.shape[0] #number of features
dim_3 = my_features.shape[1]
all_anomaly_features = np.asarray(all_anomaly_features).reshape(dim_1, dim_2, dim_3)
all_anomaly_labels = np.asarray(all_anomaly_labels).reshape(dim_1)

#check the dimentions
print(all_anomaly_features.shape)
print(all_anomaly_labels.shape)

#Merge noraml and anomaly arrays
all_data = np.concatenate((all_noraml_features,all_anomaly_features),axis=0)
all_label = np.concatenate((all_noraml_labels,all_anomaly_labels),axis=0)

#check the dimentions
print(all_data.shape)

#endocding labels
my_ec = LabelEncoder()
all_label = my_ec.fit_transform(all_label)

my_ec.transform(["normal"])
my_ec.inverse_transform(all_label)

#split data into train. validation, and test
X_train, X_test, y_train, y_test = train_test_split(all_data,all_label,test_size=0.1,shuffle=True)
X_train, X_validation, y_train, y_validation = train_test_split(X_train,y_train,test_size=0.1,shuffle=True)

#check the dimentions
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)



#build a model
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.losses import SparseCategoricalCrossentropy

n_unit=50
n_features=X_train.shape[1]
dim3= X_train.shape[2]

num_classes=2

'''
model = Sequential()
model.add(LSTM(n_unit, input_shape=(n_features,dim3)))
model.add(Dropout(0.1)) #recommnedded rate is 0.1
#model.add(Dense(1, activation="sigmoid"))
model.add(Dense(num_classes, activation = "softmax"))
model.compile(optimizer=keras.optimizers.Adam(1e-4),loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.summary()
'''
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
'''
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(n_features,dim3), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(dim3))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()

'''
#defining AE-LSTM
model = Sequential()
model.add(LSTM(n_unit, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(X_train.shape[1])) #Repeats the input n times, as we want the output shape to be (30,1)
model.add(LSTM(n_unit, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(X_train.shape[2]))) #applies a specific layer such as Dense to every sample it receives as an input.
#model.compile(optimizer=keras.optimizers.Adam(1e-4),loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
 
model.compile(optimizer='adam', loss='mae' , metrics=["accuracy"])
model.summary()

'''
model = Sequential()
model.add(LSTM(128, input_shape=(n_features,dim3)))
model.add(Dropout(rate=0.2))
model.add(RepeatVector(dim3)) #Repeats the input n times, as we want the output shape to be (30,1)
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(n_features))) #applies a specific layer such as Dense to every sample it receives as an input. 
model.compile(optimizer='adam', loss='mae')
model.summary()
'''
#train a model

#history=model.fit(X_train, y_train, validation_data=(X_validation, y_validation),epochs = 50, batch_size = 128, verbose = 1)

#history=model.fit(X_train, X_train, epochs = 50, batch_size = 128, verbose = 1)
history=model.fit(X_train, X_train, validation_data=(X_validation, X_validation),epochs = 50, batch_size = 128, verbose = 1)

#history.summary()

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'][1:43])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#plot confusion matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error


pred = model.predict(X_train)
y_pred = [0 if y[0]>y[1] else 1 for y in pred]

cm=confusion_matrix(y_train,y_pred)
print("train set")
print("0:%s - 1:%s"%(my_ec.inverse_transform([0])[0],my_ec.inverse_transform([1])[0]))
print(cm)

accuracy = (cm[0,0]+cm[1,1])/cm.sum()
precision= (cm[0,0])/(cm[0,0]+cm[1,0])
recall = (cm[0,0])/(cm[0,0]+cm[0,1])
f1=2*precision*recall/(precision+recall)

print("accuracy=%.2f f1score=%.2f"%(accuracy, f1))


pred = model.predict(X_test)
y_pred = [0 if y[0]>y[1] else 1 for y in pred]

cm=confusion_matrix(y_test,y_pred)
print("test set")
print("0:%s - 1:%s"%(my_ec.inverse_transform([0])[0],my_ec.inverse_transform([1])[0]))
print(cm)


accuracy = (cm[0,0]+cm[1,1])/cm.sum() #tp+tn/sum
precision= (cm[0,0])/(cm[0,0]+cm[1,0]) #tp/tp+fp
recall = (cm[0,0])/(cm[0,0]+cm[0,1]) #tp/tp+fn
f1=2*precision*recall/(precision+recall)

print("accuracy=%.2f f1score=%.2f"%(accuracy, f1))




'''
    return model
### Define kFOLD validation ###
accuracies = []
folds = np.array(['normal','anomaly'])
load_dir = "audio/"
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(folds):
    x_train, y_train = [], []
    for ind in train_index:
        # read features or segments of an audio file
        train_data = np.load("{0}/{1}.npz".format(load_dir,folds[ind]), 
                       allow_pickle=True)
        # for training stack all the segments so that they are treated as an example/instance
        features = np.concatenate(train_data["features"], axis=0) 
        labels = np.concatenate(train_data["labels"], axis=0)
        x_train.append(features)
        y_train.append(labels)
    # stack x,y pairs of all training folds 
    x_train = np.concatenate(x_train, axis = 0).astype(np.float32)
    y_train = np.concatenate(y_train, axis = 0).astype(np.float32)
    
    # for testing we will make predictions on each segment and average them to 
    # produce signle label for an entire sound clip.
    test_data = np.load("{0}/{1}.npz".format(load_dir,
                   folds[test_index][0]), allow_pickle=True)
    x_test = test_data["features"]
    y_test = test_data["labels"]

    model = get_network()
    model.fit(x_train, y_train, epochs = 30, batch_size = 24, verbose = 0)
    
    # evaluate on test set/fold
    y_true, y_pred = [], []
    for x, y in zip(x_test, y_test):
        # average predictions over segments of a sound clip
        avg_p = np.argmax(np.mean(model.predict(x), axis = 0))
        y_pred.append(avg_p) 
        # pick single label via np.unique for a sound clip
        y_true.append(np.unique(y)[0]) 
    accuracies.append(accuracy_score(y_true, y_pred))    
print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))


#confusion matrix result
#model = LogisticRegression()
#model = model.fit(matrix, labels)
#pred = model.predict(test_matrix)
#cm=metrics.confusion_matrix(test_labels,pred)
#print(cm)
#plt.imshow(cm, cmap='binary')
'''

