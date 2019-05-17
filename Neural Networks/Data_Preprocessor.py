#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:50:13 2019

@author: schmidt
"""

import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import math



#X_Train=pd.read_csv('house-prices-advanced-regression-techniques/train.csv').values


def string_to_unique_int(arr):
    le = preprocessing.LabelEncoder()
    le.fit(arr)
    return le.transform(arr)

def unique_int_to_one_Hot(arr,sk = True):
    L=len(arr)
#    start_time = time.time()
    if sk:
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded=arr.reshape(L,1)
        ret = onehot_encoder.fit_transform(integer_encoded)
    else:
        N=len(np.unique(arr))
        ret = np.zeros((L,N))
        for i in range(L):
            ret[i,arr[i]]=1
            
#    end_time = time.time()
#    print("%.10f seconds" % (end_time - start_time))
    return ret.T

def un_nan(arr, nan_to = 0 ):
    for n in range(len(arr)):
        if type(arr[n])!=str and np.isnan(arr[n]):
            arr[n]=nan_to 
    return arr

def valid_check(arr,max_one_hot = 5):
    str_inside,nan_inside = False,False
    
    for i in arr:
        if type(i) == str:
            str_inside = True
        
        elif math.isnan(i):
            nan_inside = True

    if nan_inside:   
        if str_inside:
            un_nan(arr,nan_to="nan")
        else:
            un_nan(arr)
        nan_inside=False
      
    if str_inside:
        return unique_int_to_one_Hot(string_to_unique_int(arr))        
        
    if str_inside==False and nan_inside==False: 
        if len(np.unique(arr)) < max_one_hot:
            return unique_int_to_one_Hot(string_to_unique_int(arr))
        else:
            return (arr-np.mean(arr))/np.std(arr)
        

def preprocess(data):
    R,C = data.shape
    processed_data = np.arange(R)
    
    for i in range(C):
        arr = data[:,i]
        arr=valid_check(arr)
        processed_data=  np.vstack((processed_data,arr))
    return unobject(processed_data.T)

def data_splitter(data,target,dist=[60,20,20]):
    dist = np.array(dist)
    DATA = np.hstack((data,target))
    dist = np.round(dist*(len(DATA)/100))
    
    np.random.shuffle(DATA)
    Train = DATA[:int(dist[0])]
    Valid = DATA[int(dist[0]):int(dist[0])+int(dist[1])]
    Test = DATA[int(dist[0])+int(dist[1]):]
    return [(Train[:,:-1],Train[:,-1].reshape(int(dist[0]),1)),((Valid[:,:-1],Valid[:,-1].reshape(int(dist[1]),1))),(Test[:,:-1],Test[:,-1].reshape(int(dist[2]),1))]
    

def unobject(data):
    R,C = data.shape
    ret = np.zeros((R,C))
    for r in range(R):
        for c in range(C):
            ret[r,c]=float(data[r,c])
    return ret
        
#Y_Train=X_Train[:,-1].reshape((len(X_Train),1))
#Ymean=np.mean(Y_Train)
#Y_Train*=1/Ymean
#X_Train=np.delete(X_Train,-1,1)
#
#X = preprocess(X_Train)
#DATA = data_splitter(X,Y_Train)
#
#Train=DATA[0]
#Y_Train = unobject(Train[1])
#X_Train = unobject(Train[0])
#
#Valid=DATA[1]
#Y_Valid = unobject(Valid[1])
#X_Valid = unobject(Valid[0])
#
#Test=DATA[2]
#Y_Test = unobject(Test[1])
#X_Test = unobject(Test[0])
    


    