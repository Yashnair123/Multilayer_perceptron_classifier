#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 02:36:53 2018

@author: Yash Nair and Raluca Vlad
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
from keras.layers import Dense, Embedding, LSTM, Dropout
import multiprocessing


def main(data_taylor, data_bach):
    train_input = data_taylor + data_bach

    train_output = []
    for _ in data_taylor:
       train_output.append(0)
    for _ in data_bach:
       train_output.append(1)
       
    data_bach_test = load('bach','1')
    data_taylor_test = load('taylor','3') + load('taylor','6') + load('taylor','7')
    
    test_input = data_bach_test + data_taylor_test
    
    test_output = []
    for _ in data_bach_test:
       test_output.append(1)
    for _ in data_taylor_test:
       test_output.append(0)
    
    x_train = np.array(train_input, dtype=float)
    y_train = np.array(train_output, dtype=float)
    x_test = np.array(test_input, dtype=float)
    y_test = np.array(test_output, dtype=float)
    
    model = Sequential()
    model.add(Dense(500, activation='relu', input_dim=len(x_train[0])))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=5, batch_size=3)
    score, accuracy = model.evaluate(x_test, y_test, batch_size=3)
    print("*****************************")
    print ("Classification accuracy: " + str(100.0 * accuracy) + "%")
    print("*****************************")

def load(composer, i):
   seg_data = []
   file_name = './Music/' + composer + i + '.wav'

   duration = librosa.get_duration(filename = file_name) - 10

   t = 10
   while t + 5 <= duration:
       data, sampling_rate = librosa.load(file_name, offset = t, duration = 5)
       seg_data.append(data[0::20])
       t += 5
   return seg_data


def load_taylor(return_dict):
    return_dict[0] = load('taylor','1') + load('taylor','2') + load('taylor','4') + load('taylor','5') + load('taylor', '8') + load('taylor', '9') + load('taylor', '10') + load('taylor', '11')


def load_bach(return_dict):
    return_dict[1] = load('bach','2') + load('bach','3') + load('bach', '4') + load('bach', '5')


if __name__ == '__main__':
    # Using multiprocessing and parallelism to improve speed
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    p = multiprocessing.Process(target=load_taylor, args=(return_dict,))
    jobs.append(p)
    p.start()
    q = multiprocessing.Process(target=load_bach, args=(return_dict,))
    jobs.append(q)
    q.start()

    for proc in jobs:
        proc.join()
    main(return_dict[0],return_dict[1])
