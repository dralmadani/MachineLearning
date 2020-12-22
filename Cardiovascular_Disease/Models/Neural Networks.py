# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:52:09 2020

@author: CZ_KAI
"""

#%% Package
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers 
from tensorflow.keras.layers import Flatten
import tensorflow.keras.backend as K
from tensorflow.keras import layers 
from keras.layers import merge, Convolution2D, Input
from keras.callbacks import Callback
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import itertools
from scipy.io import loadmat, savemat
import tensorflow as tf
from scipy import stats
import statsmodels.api as sm
#%% Random state for NN 
SEED = 123456
import os
import random as rn
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
rn.seed(SEED)

#%% Import train dataset
# Change your own dictionary where dataset located
dr = "D:\\WPI\\2020 Fall session\\CS 539\\Project\\"
Cardio_Profile = loadmat(dr+'cadio_filter.mat')
Cardio_raw = Cardio_Profile['cardiotrain']
# split triandata:x and result:y
# % Features:
# % 0 ID    
# % 1 Age | Objective Feature | age | int (days)
# % 2 Gender | Objective Feature | gender | categorical code | 1 - women, 2 - men
# % 3 Height | Objective Feature | height | int (cm) |
# % 4 Weight | Objective Feature | weight | float (kg) | 
# % 5 Systolic blood pressure | Examination Feature | ap_hi | int |
# % 6 Diastolic blood pressure | Examination Feature | ap_lo | int |
# % 7 Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# % 8 Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# % 9 Smoking | Subjective Feature | smoke | binary |
# % 10 Alcohol intake | Subjective Feature | alco | binary |
# % 11 Physical activity | Subjective Feature | active | binary |
# % 12 Presence or absence of cardiovascular disease | Target Variable | cardio | binary |
x = Cardio_raw [:,[1,2,3,4,5,6,7,8,9,10,11]]
y = Cardio_raw [:,12]
#%% Split train and test data
# use random_state to make a fixed train and test dataset
x_train,x_test, y_Train, y_test = train_test_split(x,y,test_size=0.2,random_state=555)
#%% Normalization 
# Normalization is necessary for cnn
x_train = x_train/x_train.max(axis=0)
x_test = x_test/x_test.max(axis=0) 
#%% Data Processing
# Reshape dataset for CNN training
data_array = x_train.reshape(int(len(x_train)), x_train.shape[1]) # train data
# x_test = x_test.reshape(int(len(x_test)), x_test.shape[1],  1)
#%% Split test and validation data with 10 fold Crossvalidation(CV)
I = np.array(range(0, len(data_array))).astype('int')
x_train ,x_val = train_test_split(I,test_size=0.1)
k_fold = 10
I_fold = [-1]*k_fold

for i in np.array(range(0,k_fold)):
    if len(I)<round(len(data_array)/10):
        a = np.array(range(0,len(I)))
    else:
        a = np.random.choice(len(I), round(len(data_array)/10), replace=False)
    
    I_fold[i] = I[a].astype('int')
    I = np.delete(I,a)

#%% Fully connect NN_Model 
predicted = np.array([[0]*y_Train.shape[0]]).astype('float64')
test_acc = []
test_loss = []
test_sen = []
test_pre = []
test_auc = []
I_all = np.array(range(0, len(data_array))).astype('int')
for i in range(0,k_fold):
    
    I_train = np.delete(I_all,np.argwhere(np.in1d(I_all,I_fold[i])))
    I = np.array(I_fold[i])
    # train
    x_train = data_array[I_train,:]
    y_train = y_Train[I_train]
    # validation
    x_val = data_array [I,:]
    y_val = y_Train[I]
    # test     
    # x_test 
    # y_test 
    
    # Declare model
    model = Sequential()
    model.add(Dense(10,activation='relu',input_dim=11))
    model.add(Dense(5,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    # optimizer 
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, 
                                    beta_2=0.999,decay=0.000005)
    model.compile(optimizer= 
                   adam,
                   # tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    ## setting weights for train
    weights = 1.5
    sample_weights = np.ones(shape=(len(y_train),))
    positive_train = np.count_nonzero(y_train == 1)
    t = y_train == 1
    tt = t.reshape(len(y_train))
    sample_weights[tt == 1] = (len(y_train)-positive_train)*weights/positive_train

    # 10 fold cross validation
    history = model.fit(x_train, y_train, 
                        validation_data = (x_val, y_val), 
                        epochs=2000, 
                        sample_weight=sample_weights,  
                        batch_size=1024)

	# evaluate the acc and loss with 10% test data
    scores = model.evaluate(x_test,y_test)
    test_acc.append(scores[1]*100)
    test_loss.append(scores[0]*100)
    print("test %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # print("test %s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    
    # history    
    histo = history.history
    test_predicted = model.predict(x_test)
    val_predicted = model.predict(x_val)
    labels_predict = (test_predicted > 0.5).astype(np.int)
    cm = confusion_matrix(y_test,labels_predict)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    test_Sensitivity = (TP / float(TP + FN))
    test_Precision = (TP / float(TP + FP))
    roc_test = roc_auc_score(y_test, labels_predict)
    print ('Sensitivity',test_Sensitivity)
    print('Precision',test_Precision)
    print ('ROC', roc_test)
    test_sen.append(test_Sensitivity*100)
    test_pre.append(test_Precision*100)
    
#%% Visualization & Summary 
#% 0. print average test_acc & test_loss  
print("avg acc %.2f%% (+/- %.2f%%)" % (np.mean(test_acc), np.std(test_acc))) 
print("avg sensitivity %.2f%% (+/- %.2f%%)" % (np.mean(test_sen), np.std(test_sen))) 
print("avg precision %.2f%% (+/- %.2f%%)" % (np.mean(test_pre), np.std(test_pre))) 
print("avg roc %.2f%% (+/- %.2f%%)" % (np.mean(roc_test), np.std(roc_test))) 
#% 1. Confusion Matrix
labels_predict = (test_predicted > 0.5).astype(np.int)
# labels_val = (val_predicted > 0.47).astype(np.int)
roc_test = round(roc_auc_score(y_test, labels_predict),3)

# plot confusion matrix
def cm_plot (y_test,labels_predict):
    # cm = confusion_matrix(test,predict)
    cmt = confusion_matrix(y_test,labels_predict)
    classes= ["True", "False"]  
    TP = cmt[1][1]
    TN = cmt[0][0]
    cmt [1][1] = TN
    cmt [0][0] = TP 
    normalize=False
    fig, ax = plt.subplots()
    plt.imshow(cmt, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,fontsize=15)
    plt.yticks(tick_marks, classes,fontsize=15)
    ax.xaxis.tick_top()
    fmt = '.2f' if normalize else 'd'
    thresh = cmt.max() / 2.
    for i, j in itertools.product(range(cmt.shape[0]), range(cmt.shape[1])):
              plt.text(j, i, format(cmt[i, j], fmt),
              horizontalalignment="center",
              color="white" if cmt[i, j] > thresh else "black",
              fontsize=17)
    
    plt.tight_layout()
    plt.title('Actual Cardio', fontsize=18)
    plt.ylabel('Predicted Cardio', fontsize=18)
    # plt.xlabel('Actual impact')
    # plt.figure(dpi = 600)
    plt.figure(figsize=(40, 40))
    plt.show()
    return cmt
# plot confusion matrix
cm_plot (y_test,labels_predict)
# Evaluation
# cm = confusion_matrix(y_test,labels_predict)
TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
test_Accuracy = round((float (TP+TN) / float(TP + TN + FP + FN)),3)
test_Sensitivity = round((TP / float(TP + FN)),3)
test_Specificity = round((TN / float(TN + FP)),3)
test_Precision = round((TP / float(TP + FP)),3)
test_F1_score = round(2 * ((test_Precision * test_Sensitivity) / (test_Precision + test_Sensitivity)),3)

#add cm tabel at bottom
cell = np.asarray([test_Accuracy, test_Sensitivity, test_Specificity, test_Precision, test_F1_score,roc_test ]).transpose()
cell = cell.reshape(6,1)
columns = (['Accuracy','Sensitivity','Specificity','Precision','F1_score','AUC'])

# plt.figure(figsize=(5, 1))
plt.figure(dpi=1200)
plt.axis('tight')
plt.axis('off')
ytable=plt.table(cellText=cell,rowLabels=columns,loc='center',colWidths=[.5]*2)
ytable.set_fontsize(34)
ytable.scale(1, 6)
plt.show()

#%% Validation Acc
plt.figure(2)
plt.figure(dpi=1200)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#%% Validation Loss
plt.figure(3)
plt.figure(dpi=1200)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()