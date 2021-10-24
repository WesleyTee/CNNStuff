# -*- coding: utf-8 -*-
"""
Created on Thu May 20 04:32:35 2021

@author: Wesley Tee
"""

## Testing Lucas CNN by Felix Reise

#import modules
import os
import glob
#import sys
import pandas as pd
from lucas_classification import lucas_classification
import cnn_models as cnn 
from tensorflow.keras.models import Model



############################# Data loading ###################################

#Presorted small subsection of data in sortcnndata.mlx 

x_trainave = pd.read_csv("x_train256.csv",header=None)
x_trainold = pd.read_csv("x_train.csv")
print('X_train loaded')
x_valave = pd.read_csv("x_val256.csv",header=None)
print('x_val loaded')
x_testave = pd.read_csv("x_test256.csv",header=None)
y_train_k = pd.read_csv("y_train_k.csv")
y_train_n = pd.read_csv("y_train_n.csv")
y_train_p = pd.read_csv("y_train_p.csv")
print('Y train values loaded')
y_val_k = pd.read_csv("y_val_k.csv")
y_val_n = pd.read_csv("y_val_n.csv")
y_val_p = pd.read_csv("y_val_p.csv")
print('Y validate values loaded')
y_test_k = pd.read_csv("y_test_k.csv")
y_test_n = pd.read_csv("y_test_n.csv")
y_test_p = pd.read_csv("y_test_p.csv")

hsitest = pd.read_csv("HSI_256_Average.csv",header=None)
#Time to take HSI_256_PCA
#hsitest = pd.read_csv("HSI_256_PCA.csv",header=None)
#HSI PCA 128
#hsitest = pd.read_csv("HSI_128_PCA.csv",header=None)
#HSI 48 PCA
#hsitest = pd.read_csv("HSI_48_PCA.csv",header=None)
#PCA 12
#hsitest = pd.read_csv("HSI_12_PCA.csv",header=None)
paramvalold = pd.read_csv("Param_Data.csv",header=None)

#using new values. drop middle column

paramval = paramvalold.drop([1],axis=1)
#paramval = paramval.drop([0],axis=1)
#paramval = paramval.drop([2],axis=1)
##Join the y_train vals
#combtrain1 = y_train_k.join(y_train_n)
#y_train = combtrain1.join(y_train_p)
#
##Join the y vals
#combtrain2 = y_val_k.join(y_val_n)
#y_val = combtrain2.join(y_val_p)
#
##join test vals
#combtrain3 = y_test_k.join(y_test_n)
#y_test = combtrain3.join(y_test_p)

#Param1 = 3.189
#Param2 = 48.195


############## Using NEW VALUES 
########### Divide data 

#70% for train
x_train = hsitest.head(12291)
#15 % for val
x_val = hsitest.loc[12292:14925]
#15%for test
x_test = hsitest.loc[14926:17559]

#divide data for y
y_train = paramval.head(12291)
y_val = paramval.loc[12292:14925]
y_test = paramval.loc[14926:17559]

Model.summary(cnn.getKerasModel('LucasCNN'))
cnn_model = cnn.getKerasModel('LucasCNN')
#cnn_model.predict(x_test[:10])
######################## Get CNN values
##
# data is separated into hsi and corresponding topsoil data
# x_train is large set of hsi data 
# x_val is data to validate. smaller set of 
# separate model for each y value, 1 for Nitrogen, 1 for K, 1 for carbon

#Testing K
score_k = lucas_classification(
    data=[x_train, x_val,x_test, y_train, y_val,y_test],
    model_name="LucasCNN",
    batch_size=128,
    epochs=200,
    random_state=42)

print(score_k)


#compile_and_fit('LucasCNN')
#
#def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
#  if optimizer is None:
#    optimizer = get_optimizer()
#  model.compile(optimizer=optimizer,
#                loss=tf.keras.losses.mean_squared_error(from_logits=True),
#                metrics = ["accuracy"]
#
#  #model.summary()
#
#  history = model.fit(
#    train_ds,
#    steps_per_epoch = STEPS_PER_EPOCH,
#    epochs=max_epochs,
#    validation_data=validate_ds,
#    callbacks=get_callbacks(name),
#    verbose=0)
#  
#  return history


############################# CODE GRAVEYARD ################################

# ##get data
# #set path
# os.chdir('C:\\Users\\psite\\Documents\\MATLAB\\ENN541\\LUCAS2015_Soil_Spectra_EU28')
# data = pd.concat(map(pd.read_csv, glob.glob('*.csv')))
# print('Data acquired')

# #get the lucas topsoil data
# #NOTE: I edited the category Point_ID to PointID to match the csv layout
# topsoil = pd.read_excel(r'C:\Users\psite\Documents\MATLAB\ENN541\LUCAS_Topsoil_2015_20200323.xlsx')
# print('Topsoil data acquired')

# #set back to main folder
# os.chdir('C:\\Users\\psite\\Documents\\ENN541\\CNN-SoilTextureClassification-master')

# #Merge tables together
# data_comb = pd.merge(data,topsoil, how='inner')
# print('Merge complete')

### DO NOT DO THIS. MAKES A 1.8GB FILE
# #save the output csv to read
# data_comb.to_csv(r'C:\Users\psite\Documents\ENN541\CNN-SoilTextureClassification-master\testdata.csv')
# print('csv made')