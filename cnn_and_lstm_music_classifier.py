#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:42:10 2018

@author: yashnair
"""

import os
import pandas as pd
import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv2D, Flatten
import multiprocessing


def main(data_taylor, data_bach, data_elec):
    data_taylor_test = load('taylor',23) + load('taylor',24) + load('taylor',25)
    data_bach_test = load('bach',10)
    data_elec_test = load('elec', 4) + load('elec', 5)
    
    
    train_input = data_taylor + data_bach
    
    train_output = []
    for _ in data_taylor:
        train_output.append(0)
    for _ in data_bach:
        train_output.append(1)
    
    test_input = data_taylor_test + data_bach_test
    
    test_output = []
    for _ in data_taylor_test:
        test_output.append(0)
    for _ in data_bach_test:
        test_output.append(1)
    
    train_input = data_elec + data_bach
    
    train_output = []
    for _ in data_elec:
        train_output.append(0)
    for _ in data_bach:
        train_output.append(1)
    
    test_input = data_elec_test + data_bach_test
    
    test_output = []
    for _ in data_elec_test:
        test_output.append(0)
    for _ in data_bach_test:
        test_output.append(1)
    
    
    # Multilayer perceptron
    x_train = np.array(train_input, dtype=float)
    y_train = np.array(train_output, dtype=float)
    x_test = np.array(test_input, dtype=float)
    y_test = np.array(test_output, dtype=float)
    
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=len(x_train[0])))
    model.add(Dropout(0.6))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.38))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=4, batch_size=15)
    dictionary = history.history
    score, accuracy = model.evaluate(x_test, y_test, batch_size=15)
    print("*****************************")
    print ("Classification accuracy: " + str(100.0 * accuracy) + "%")
    print("*****************************")
    
    
    # RNN
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(350, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.45))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.45))
    model.add(LSTM(10, return_sequences=False))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=3, batch_size=10)
    score, accuracy = model.evaluate(x_test, y_test, batch_size=10)
    print("*****************************")
    print ("Classification accuracy: " + str(100.0 * accuracy) + "%")
    print("*****************************")
    
    
    # CNN
    x_train_matrix = []
    for seq in range(len(train_input)):
        s = []
        for i in range(74):
            s.append(train_input[seq][i*74:(i+1)*74])
        x_train_matrix.append(s)
    
    x_test_matrix = []
    for seq in range(len(test_input)):
        s = []
        for i in range(74):
            s.append(test_input[seq][i*74:(i+1)*74])
        x_test_matrix.append(s)
    
    x_train = np.array(x_train_matrix, dtype=float)
    y_train = np.array(train_output, dtype=float)
    x_test = np.array(x_test_matrix, dtype=float)
    y_test = np.array(test_output, dtype=float)
    
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
    
    model = Sequential()
    model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(74,74,1)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=1, batch_size=20)
    score, accuracy = model.evaluate(x_test, y_test, batch_size=20)
    print("*****************************")
    print ("Classification accuracy: " + str(100.0 * accuracy) + "%")
    print("*****************************")
    
def load(composer, i):
    seg_data = []
    file_name = '../Music/' + composer + str(i) + '.wav'

    duration = librosa.get_duration(filename = file_name) - 10

    t = 10
    while t + 5 <= duration:
        data, sampling_rate = librosa.load(file_name, offset = t, duration = 5)
        seg_data.append(data[0::20])
        t += 5
    return seg_data

def load_taylor(return_dict):
    data_taylor = []
    for i in range(1, 23):
        data_taylor = data_taylor + load('taylor', i)
    return_dict[0] = data_taylor

def load_bach(return_dict):
    data_bach = []
    for i in range(1, 10):
        data_bach = data_bach + load('bach', i)
    return_dict[1] = data_bach

def load_elec(return_dict):
    data_elec = []
    for i in range(1, 4):
        data_elec = data_elec + load('elec', i)
    return_dict[2] = data_elec
        
if __name__ == '__main__':
    # Using multiprocessing and parallelism to improve speed
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    p1 = multiprocessing.Process(target=load_taylor, args=(return_dict,))
    jobs.append(p1)
    p1.start()
    p2 = multiprocessing.Process(target=load_bach, args=(return_dict,))
    jobs.append(p2)
    p2.start()
    p3 = multiprocessing.Process(target=load_elec, args=(return_dict,))
    jobs.append(p3)
    p3.start()

    for proc in jobs:
        proc.join()
    main(return_dict[0],return_dict[1], return_dict[2])
