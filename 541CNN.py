# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:37:50 2021

@author: psite
"""


import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import os, sys

from tensorflow.keras.layers import (Concatenate, Conv1D, Dense, Flatten,
                                     Input, MaxPooling1D, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras import models
from matplotlib import pyplot

def HsiEstimator(input_dim, output_dim, use_max_pooling=True):

    seq_length = input_dim
    kernel_size = 3
    activation = "relu"
    padding = "valid"

    inp = Input(shape=(seq_length, 1))

    # CONV1
    x = Conv1D(filters=32,
               kernel_size=kernel_size,
               activation=activation,
               padding=padding)(inp)
    
    if use_max_pooling: x = MaxPooling1D(2)(x)

    # CONV2
    x = Conv1D(filters=32,
               kernel_size=kernel_size,
               activation=activation,
               padding=padding)(x)
    if use_max_pooling: x = MaxPooling1D(2)(x)

    # CONV3
    x = Conv1D(filters=64,
               kernel_size=kernel_size,
               activation=activation,
               padding=padding)(x)
    if use_max_pooling: x = MaxPooling1D(2)(x)

    # CONV4
    x = Conv1D(filters=64,
               kernel_size=kernel_size,
               activation=activation,
               padding=padding)(x)
    if use_max_pooling: x = MaxPooling1D(2)(x)

    # Flatten, FC1, FC2, Softmax
    x = Flatten()(x)
    x = Dense(120, activation=activation)(x)
    x = Dense(160, activation=activation)(x)
    x = Dense(output_dim, activation=None)(x)

    return Model(inputs=inp, outputs=x)

def parameter_rmse(pred, actual):
    print("Nitrogen RMSE", ((1/actual[0].shape[0])*sum((actual[0] - pred[:, 0])**2))**0.5)
    print("Phosphorus RMSE", ((1/actual[1].shape[0])*sum((actual[1] - pred[:, 1])**2))**0.5)

def split_data(hsi_data_path, param_data_path):
    x_train = pd.read_csv(hsi_data_path, header=None)
    y_train = pd.read_csv(param_data_path, header=None)

    train_set_size = int(0.8 * x_train.shape[0])
    test_set_size = x_train.shape[0] - train_set_size

    training_ind = np.random.permutation(x_train.shape[0])
    test_ind = training_ind[-test_set_size:]

    x_train = x_train[:train_set_size]
    x_test = x_train[-test_set_size:]

    y_train = y_train[:train_set_size]
    y_test = y_train[-test_set_size:]
    
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    return x_train, x_test, y_train, y_test

def generate_dual_model(model_name, batch_size, epochs, use_max_pooling=True):
    if model_name == "model_ref":
        print("Training model_ref:")
        model = HsiEstimator(262, 2, True)
        #print(x_train)
        model.summary()
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_256_Average.csv", "Param_Data.csv")
        #print(x_train)
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_256":
        print("Training model_256:")
        model = HsiEstimator(256, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_256_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data = (x_test, y_test),
        )
#        pyplot.title('RMSE Over Epoch for PCA 12 Values')
#        pyplot.plot(history.history['root_mean_squared_error'], label='Training')
#        pyplot.plot(history.history['val_root_mean_squared_error'], label='Validation')
#        pyplot.legend()
#        pyplot.show()
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_128":
        print("Training model_128:")
        model = HsiEstimator(128, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_128_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_48":
        print("Training model_48:")
        model = HsiEstimator(48, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_48_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_12":
        print("Training model_12:")
        model = HsiEstimator(12, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_12_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_8":
        print("Training model_8:")
        model = HsiEstimator(8, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_8_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_4":
        print("Training model_4:")
        model = HsiEstimator(4, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_4_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_2":
        print("Training model_2:")
        model = HsiEstimator(2, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_2_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    if model_name == "model_1":
        print("Training model_1:")
        model = HsiEstimator(1, 2, False)
        model.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train, x_test, y_train, y_test = split_data("HSI_1_PCA.csv", "Param_Data.csv")
        model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1)
        return model, x_train, x_test, y_train, y_test
    
    print("No such model exists")
    return None

def generate_single_models(model_name, batch_size, epochs, use_max_pooling=True):
    if model_name == "model_ref":
        print("Training model_ref_N:")
        model_N = HsiEstimator(262, 1, False)
        print("HSI Estimation Complete")
        #earlystopping = EarlyStopping(monitor="loss", patience=10)

        print("Compilation Complete")
        x_train_N, x_test_N, y_train_N, y_test_N = split_data("HSI_256_Average.csv", "Param_Data_N_only.csv")
        print("Data Loaded")        
        model_N.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                #validation_data=(x_test_N, y_test_N)
                )
        history256refN = model_N.fit(
                x_train_N, y_train_N,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_N, y_test_N),
                verbose=0)
        pyplot.title('RMSE Over Epoch for Reference Model N')
        pyplot.plot(history256refN.history['root_mean_squared_error'], label='Training')
        pyplot.plot(history256refN.history['val_root_mean_squared_error'], label='Test')
        pyplot.xlabel("Number of Epochs")
        pyplot.ylabel("Root Mean Square Error")
        pyplot.grid()
        pyplot.legend()
        pyplot.show()
        
        print("Fitting Done")
        print("Training model_ref_P:")
        x_train_P, x_test_P, y_train_P, y_test_P = split_data("HSI_256_Average.csv", "Param_Data_P_only.csv")
        model_P = HsiEstimator(262, 1, False)
        model_P.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
                #validation_data=(x_test_P, y_test_P))
                )
        history256refP = model_P.fit(
                x_train_P, y_train_P,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_P, y_test_P),
                verbose=0)
        
        pyplot.title('RMSE Over Epoch for Reference Model P')
        pyplot.plot(history256refP.history['root_mean_squared_error'], label='Training')
        pyplot.plot(history256refP.history['val_root_mean_squared_error'], label='Test')
        pyplot.xlabel("Number of Epochs")
        pyplot.ylabel("Root Mean Square Error")
        pyplot.grid()
        pyplot.legend()
        pyplot.show()
        


        #Evaluate model
        _,train_score_N = model_N.evaluate(x_train_N, y_train_N, batch_size=batch_size)
        _,score_N = model_N.evaluate(x_test_N, y_test_N, batch_size=batch_size)
        print('Reference Model 256 N - Training Error: %.3f, Test Error: %.3f' % (train_score_N, score_N))
        
        _,train_score_P = model_P.evaluate(x_train_P, y_train_P, batch_size=batch_size)
        _,score_P = model_P.evaluate(x_test_P, y_test_P, batch_size=batch_size)
        print('Reference Model 256 P - Training Error: %.3f, Test Error: %.3f' % (train_score_P, score_P))        
        
        return model_N, model_P, x_train_N, x_train_P, x_test_N, x_test_P, y_train_N, y_train_P, y_test_N, y_test_P
    
    if model_name == "model_256":
        ppe
        print("Training model_256_N:")
        model_N = HsiEstimator(256, 1, False)
        model_N.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_N, x_test_N, y_train_N, y_test_N = split_data("HSI_256_PCA.csv", "Param_Data_N_only.csv")
        historypca256N = model_N.fit(
                x_train_N, y_train_N,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_N, y_test_N),
                verbose=0)
        print("Training model_256_P:")
        model_P = HsiEstimator(256, 1, False)
        historypca256P = model_P.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_P, x_test_P, y_train_P, y_test_P = split_data("HSI_256_PCA.csv", "Param_Data_P_only.csv")
        historypca256P = model_P.fit(
                x_train_P, y_train_P,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_P, y_test_P),
                verbose=0)
#        pyplot.title('RMSE Over Epoch for Reference Model P')
#        pyplot.plot(history256refN.history['root_mean_squared_error'], label='Model 1')
#        pyplot.plot(history256refP.history['root_mean_squared_error'], label='Model 2')
#        pyplot.legend()
#        pyplot.show()
        _,train_score_N = model_N.evaluate(x_train_N, y_train_N, batch_size=batch_size)
        _,score_N = model_N.evaluate(x_test_N, y_test_N, batch_size=batch_size)
        print('PCA Model 256 N - Training Error: %.3f, Test Error: %.3f' % (train_score_N, score_N))

        _,train_score_P = model_P.evaluate(x_train_P, y_train_P, batch_size=batch_size)
        _,score_P = model_P.evaluate(x_test_P, y_test_P, batch_size=batch_size)
        print('PCA Model 256 P - Training Error: %.3f, Test Error: %.3f' % (train_score_P, score_P)) 
        
        return model_N, model_P, x_train_N, x_train_P, x_test_N, x_test_P, y_train_N, y_train_P, y_test_N, y_test_P
    
    if model_name == "model_128":
        print("Training model_128_N:")
        model_N = HsiEstimator(128, 1, False)
        model_N.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_N, x_test_N, y_train_N, y_test_N = split_data("HSI_128_PCA.csv", "Param_Data_N_only.csv")
        historypca128N = model_N.fit(
                x_train_N, y_train_N,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_N, y_test_N),
                verbose=0)
        print("Training model_128_P:")
        model_P = HsiEstimator(128, 1, False)
        model_P.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_P, x_test_P, y_train_P, y_test_P = split_data("HSI_128_PCA.csv", "Param_Data_P_only.csv")
        historypca128P = model_P.fit(
                x_train_P, y_train_P,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_P, y_test_P),
                verbose=0)
        
        _,train_score_N = model_N.evaluate(x_train_N, y_train_N, batch_size=batch_size)
        _,score_N = model_N.evaluate(x_test_N, y_test_N, batch_size=batch_size)
        print('PCA Model 128 N - Training Error: %.3f, Test Error: %.3f' % (train_score_N, score_N))

        _,train_score_P = model_P.evaluate(x_train_P, y_train_P, batch_size=batch_size)
        _,score_P = model_P.evaluate(x_test_P, y_test_P, batch_size=batch_size)
        print('PCA Model 128 P - Training Error: %.3f, Test Error: %.3f' % (train_score_P, score_P)) 
        
        return model_N, model_P, x_train_N, x_train_P, x_test_N, x_test_P, y_train_N, y_train_P, y_test_N, y_test_P
    
    if model_name == "model_48":
        print("Training model_48_N:")
        model_N = HsiEstimator(48, 1, False)
        model_N.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_N, x_test_N, y_train_N, y_test_N = split_data("HSI_48_PCA.csv", "Param_Data_N_only.csv")
        historypca48N = model_N.fit(
                x_train_N, y_train_N,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_N, y_test_N),
                verbose=0)
        print("Training model_48_P:")
        model_P = HsiEstimator(48, 1, False)
        model_P.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_P, x_test_P, y_train_P, y_test_P = split_data("HSI_48_PCA.csv", "Param_Data_P_only.csv")
        historypca48P = model_P.fit(
                x_train_P, y_train_P,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_P, y_test_P),
                verbose=0)
        
        _,train_score_N = model_N.evaluate(x_train_N, y_train_N, batch_size=batch_size)
        _,score_N = model_N.evaluate(x_test_N, y_test_N, batch_size=batch_size)
        print('PCA Model 48 N - Training Error: %.3f, Test Error: %.3f' % (train_score_N, score_N))

        _,train_score_P = model_P.evaluate(x_train_P, y_train_P, batch_size=batch_size)
        _,score_P = model_P.evaluate(x_test_P, y_test_P, batch_size=batch_size)
        print('PCA Model 48 P - Training Error: %.3f, Test Error: %.3f' % (train_score_P, score_P)) 
        
        return model_N, model_P, x_train_N, x_train_P, x_test_N, x_test_P, y_train_N, y_train_P, y_test_N, y_test_P
    
    if model_name == "model_12":
        print("Training model_12_N:")
        model_N = HsiEstimator(12, 1, False)
        model_N.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_N, x_test_N, y_train_N, y_test_N = split_data("HSI_12_PCA.csv", "Param_Data_N_only.csv")
        historypca12N = model_N.fit(
                x_train_N, y_train_N,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_N, y_test_N),
                verbose=0)
        print("Training model_12_P:")
        model_P = HsiEstimator(12, 1, False)
        model_P.compile(
                loss="mean_squared_error",
                optimizer="adam",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
        x_train_P, x_test_P, y_train_P, y_test_P = split_data("HSI_12_PCA.csv", "Param_Data_P_only.csv")
        historypca12P = model_P.fit(
                x_train_P, y_train_P,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test_P, y_test_P),
                verbose=0)
        
        _,train_score_N = model_N.evaluate(x_train_N, y_train_N, batch_size=batch_size)
        _,score_N = model_N.evaluate(x_test_N, y_test_N, batch_size=batch_size)
        print('PCA Model 12 N - Training Error: %.3f, Test Error: %.3f' % (train_score_N, score_N))

        _,train_score_P = model_P.evaluate(x_train_P, y_train_P, batch_size=batch_size)
        _,score_P = model_P.evaluate(x_test_P, y_test_P, batch_size=batch_size)
        print('PCA Model 12 P - Training Error: %.3f, Test Error: %.3f' % (train_score_P, score_P)) 
        
        return model_N, model_P, x_train_N, x_train_P, x_test_N, x_test_P, y_train_N, y_train_P, y_test_N, y_test_P
    print("No such model exists")
    return None

dual_model = False

batch_size = 128
epochs = 300 #just to test, normal is 300

# Model save dir
#d = os.path.join(os.getcwd(), "models")
#if not os.path.exists(d):
#    os.makedirs(d)
#    os.chmod(d, 0o444)

if dual_model:
    # Train single models for estimation of N and P
    model_ref, x_train_ref, x_test_ref, y_train_ref, y_test_ref = generate_dual_model(model_name="model_ref", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_ref.save(os.path.join(d, "model_ref_" + str(batch_size) + "_" + str(epochs) + "_dual.model"))

    model_256, x_train_256, x_test_256, y_train_256, y_test_256 = generate_dual_model(model_name="model_256", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_256.save(os.path.join(d, "model_256_" + str(batch_size) + "_" + str(epochs) + "_dual.model"))

    model_128, x_train_128, x_test_128, y_train_128, y_test_128 = generate_dual_model(model_name="model_128", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_128.save(os.path.join(d, "model_128_" + str(batch_size) + "_" + str(epochs) + "_dual.model"))

    model_48, x_train_48, x_test_48, y_train_48, y_test_48 = generate_dual_model(model_name="model_48", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_48.save(os.path.join(d, "model_48_" + str(batch_size) + "_" + str(epochs) + "_dual.model"))

    model_12, x_train_12, x_test_12, y_train_12, y_test_12 = generate_dual_model(model_name="model_12", batch_size=batch_size, epochs=epochs, use_max_pooling=False)
    #model_12.save(os.path.join(d, "model_12_" + str(batch_size) + "_" + str(epochs) + "_dual.model"))
    
else:
    # Train individual models for N and P
    model_ref_N, model_ref_P, x_train_ref_N, x_train_ref_P, x_test_ref_N, x_test_ref_P, y_train_ref_N, y_train_ref_P, y_test_ref_N, y_test_ref_P = generate_single_models(model_name="model_ref", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_ref_N.save(os.path.join(d, "model_ref_N_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    #model_ref_P.save(os.path.join(d, "model_ref_P_" + str(batch_size) + "_" + str(epochs) + "_single.model"))

    model_256_N, model_256_P, x_train_256_N, x_train_256_P, x_test_256_N, x_test_256_P, y_train_256_N, y_train_256_P, y_test_256_N, y_test_256_P = generate_single_models(model_name="model_256", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_256_N.save(os.path.join(d, "model_256_N_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    #model_256_P.save(os.path.join(d, "model_256_P_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    
    model_128_N, model_128_P, x_train_128_N, x_train_128_P, x_test_128_N, x_test_128_P, y_train_128_N, y_train_128_P, y_test_128_N, y_test_128_P = generate_single_models(model_name="model_128", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_128_N.save(os.path.join(d, "model_128_N_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    #model_128_P.save(os.path.join(d, "model_128_P_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    
    model_48_N, model_48_P, x_train_48_N, x_train_48_P, x_test_48_N, x_test_48_P, y_train_48_N, y_train_48_P, y_test_48_N, y_test_48_P = generate_single_models(model_name="model_48", batch_size=batch_size, epochs=epochs, use_max_pooling=True)
    #model_48_N.save(os.path.join(d, "model_48_N_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
   # model_48_P.save(os.path.join(d, "model_48_P_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    
    model_12_N, model_12_P, x_train_12_N, x_train_12_P, x_test_12_N, x_test_12_P, y_train_12_N, y_train_12_P, y_test_12_N, y_test_12_P = generate_single_models(model_name="model_12", batch_size=batch_size, epochs=epochs, use_max_pooling=False)
    #model_12_N.save(os.path.join(d, "model_12_N_" + str(batch_size) + "_" + str(epochs) + "_single.model"))
    #model_12_P.save(os.path.join(d, "model_12_P_" + str(batch_size) + "_" + str(epochs) + "_single.model"))

score_ref = model_ref.evaluate(x_test_ref, y_test_ref, batch_size=batch_size)
score_256 = model_256.evaluate(x_test_256, y_test_256, batch_size=batch_size)
score_128 = model_128.evaluate(x_test_128, y_test_128, batch_size=batch_size)
score_48 = model_48.evaluate(x_test_48, y_test_48, batch_size=batch_size)
score_12 = model_12.evaluate(x_test_12, y_test_12, batch_size=batch_size)

#Evaluate N
#score_ref_N = model_ref_N.evaluate(x_test_ref_N, y_test_ref_N, batch_size=batch_size)
#score_256_N = model_256_N.evaluate(x_test_256_N, y_test_256_N, batch_size=batch_size)
#score_128_N = model_128_N.evaluate(x_test_128_N, y_test_128_N, batch_size=batch_size)
#score_48_N = model_48_N.evaluate(x_test_48_N, y_test_48_N, batch_size=batch_size)
#score_12_N = model_12_N.evaluate(x_test_12_N, y_test_12_N, batch_size=batch_size)
#
##Evaluate P
#score_ref_P = model_ref_P.evaluate(x_test_ref_P, y_test_ref_P, batch_size=batch_size)
#score_256_P = model_256_P.evaluate(x_test_256_P, y_test_256_P, batch_size=batch_size)
#score_128_P = model_128_P.evaluate(x_test_128_P, y_test_128_P, batch_size=batch_size)
#score_48_P = model_48_P.evaluate(x_test_48_P, y_test_48_P, batch_size=batch_size)
#score_12_P = model_12_P.evaluate(x_test_12_P, y_test_12_P, batch_size=batch_size)
    
#predictions = model_ref.predict(x_test_ref)
#parameter_rmse(predictions, y_test)
#Plot N
#pyplot.title('Validated RMSE Across P for Single Output')
#pyplot.plot(history256refP.history['val_root_mean_squared_error'], label='Reference Model')
#pyplot.plot(historypca256P.history['val_root_mean_squared_error'], label='PCA_256')
#pyplot.plot(historypca128P.history['val_root_mean_squared_error'], label='PCA_128')
#pyplot.plot(historypca48P.history['val_root_mean_squared_error'], label='PCA_48')
#pyplot.plot(historypca12P.history['val_root_mean_squared_error'], label='PCA_12')
#pyplot.legend()
#pyplot.show()

print(y_test[:10])
predictions = model_12.predict(x_test)
print(predictions[:10])