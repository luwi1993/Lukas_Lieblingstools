#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:08:29 2019

@author: schmidt
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 

class Regression:
    
    def __init__(self,title="Test",dLayers=[10],unlinearity=[0],unlin_args=[1],bias=True):
        self.title=title
        self.dLayers = np.hstack((dLayers,[1]))
        self.unlinearity = np.hstack((unlinearity,[0]))
        self.unlin_args=    np.hstack((unlin_args,[1]))
        self.nLayers= len(self.dLayers)
        self.nWeightsets=self.nLayers-1
        self.Input_D = self.dLayers[0]
        self.bias = bias
        
        if np.size(unlin_args) == np.size(self.unlinearity):
            self.unlin_args = unlin_args
        else:
            # if no extra argument is choosen the arguments are matched to the default of the given unlinearity
            self.unlin_args = np.zeros(len(self.unlinearity))
            for Layer in range(self.nWeightsets):
                if self.unlinearity[Layer] >= 3:
                    self.unlin_args[Layer]=0
                else:
                    self.unlin_args[Layer]=1
        
    def pred(self,x,Layer=-1):
        if Layer == -1:
            Layer=self.nLayers-1
        if Layer == 0:
            return x
        else:
            y=self.pred(x,Layer-1).dot(self.get_weights(Layer-1))
            if self.bias:
                y+=self.get_bias(Layer-1)
            return self.get_unlinearity(y,Layer) 
    
    def err(self,x,t):
        y=self.pred(x)
        return np.mean((t-y)**2)
    
    def Error_rate(self,x,t):
        y=self.pred(x)
        return np.mean(np.abs(t-y)/t)

    def show_error(self,x,t,offset=0,show_n=1):
        dif=t[offset:show_n+offset]-self.pred(x[offset:show_n+offset])
        plt.bar(np.arange(len(dif))+offset,dif.flatten(),alpha=0.5) 
    
    def get_dErr(self,x,t,Layer):
            
        y = self.pred(x) 
        z = self.pred(x,Layer+1)        
            
        if Layer >= self.nWeightsets - 1:
            return -2/len(y)*(t-y*sum(t,0)) 
             
        elif Layer >=0:
            dErr = self.get_dErr(x,t,Layer+1).dot(self.get_weights(Layer+1).T)
        else: 
            print("no valid case in get_dErr()")    
        return self.get_dunlinearity(dErr,z,Layer)
    
    def fit(self,x,t,lr,up=1,it=1,show_error=False,print_progress=True,lr_ratio=1,momentum=0,Valid=0):
        
        lr_w=lr
        lr_b=lr_ratio*lr_w
        
        velocity = []
        for i in range(self.nWeightsets):
            velocity.append(0)
            
        for u in range(up):
            if u%5 == 0:
                self.show_error(x,t,0,20)
            if print_progress:
                print("Progress:",np.round(100*(u+1)/up,decimals=1),"% \t","Train_Error:",self.err(x,t))#,"pred:",self.pred(x),"Target:",t)
                if Valid:
                    print("Valid_Error:",self.err(Valid[0],Valid[1]))
            for Layer in range(self.nWeightsets-1,-1,-1):
                for i in range(it):
                  
                    weights_Gradient=self.pred(x,Layer).T.dot(self.get_dErr(x,t,Layer))
                    
                    velocity[Layer] = momentum*velocity[Layer]+weights_Gradient
                    
                    weights=self.get_weights(Layer)-lr_w*velocity[Layer]                   
                    self.safe_weights(Layer,weights)
                    
                    bias_Gradient=self.get_dErr(x,t,Layer)
                    bias = self.get_bias(Layer)-lr_b*bias_Gradient.sum(axis=0)
                    self.safe_bias(Layer,bias)
            
#    def fit(self,x,t,lr=0.1,it=1,show_error=False):
#        for i in range(it):
#            y=self.pred(x)
#            bias_Gradient = -sum(t-y)
#            self.bias = self.bias - lr*bias_Gradient
#            
#            y=self.pred(x)
#            weights_Gradient= -sum(x*(t-y),axis=0)
#            self.weights =self.weights-lr*weights_Gradient.reshape(self.Input_D,1)
#            
#            print("it",i+1,"von",it,"Progress:",np.round(100*(i+1)/it,decimals=1),"% \t","Error:",self.err(x,t))#,"pred:",self.pred(x),"Target:",t)
#            if show_error and i%np.round(it/10)==0:
#                self.show_error(x,t,0,20)
        

# _________________Unlinearitys__________________:
        
    # unlinearity Numer 0
    def linear(self,x,alpha=1):
        return alpha*x
    
    def dlinear(self,x,alpha=1):
        return alpha
    
    # unlinearity Numer 1
    def sigmoid(self,x,alpha=1):
        return 1/(1+np.exp(-alpha*x))
    
    def dsigmoid(self,x,alpha=1):
        y = self.sigmoid(x)
        return alpha*y*(1-y)

    # unlinearity Numer 2
    def softmax(self,x,alpha=1):      
        return alpha*np.exp(x)/np.sum(np.exp(x),axis = 0)       
    
    def dsoftmax(self,x,alpha=1):
        y = self.softmax(x)
        return alpha*y*(1-y)

    # unlinearity Numer 3
    def relu(self,x,alpha = 0):
        m = np.sign(x)
        n = alpha*(m-np.ones(np.shape(m)))/2
        m = (m+np.ones(np.shape(m)))/2
        return (m-n)*x
    
    def drelu(self,x,alpha=0):
        return self.heaviside(x,alpha)
   
    # unlinearity Numer 4
    def heaviside(self,x,alpha=0):
        m = np.sign(x)
        n = alpha*(m-np.ones(np.shape(m)))/2
        m = (m+np.ones(np.shape(m)))/2
        return m+n 
    
    def tanh(self,x,alpha=0):
        return np.tanh(x)
    
    def dtanh(self,x,alpha=0):
        return 1-np.tanh(x)**2

    def get_unlinearity(self,y,Layer):
        # choosable unlinearity in each Layer:            
        # 0: no unlinearity 
        if self.unlinearity[Layer-1] == 0:
            y= self.linear(y,self.unlin_args[Layer-1])              
        # 1: sigmoid 
        elif self.unlinearity[Layer-1] == 1:
            y = self.sigmoid(y,self.unlin_args[Layer-1])        
        # 2: softmax 
        elif self.unlinearity[Layer-1] == 2:
            y = self.softmax(y,self.unlin_args[Layer-1])   
        # 3: relu
        elif self.unlinearity[Layer-1] == 3:
            y = self.relu(y,self.unlin_args[Layer-1])
        # 4: heaviside
        elif self.unlinearity[Layer-1] == 4:
            y = self.heaviside(y,self.unlin_args[Layer-1])
        # 5: tanh
        elif self.unlinearity[Layer-1] == 5:
            y = self.tanh(y,self.unlin_args[Layer-1]) 
        return y
    
    def get_dunlinearity(self,dErr,z,Layer):
        if self.unlinearity[Layer] == 0:
            dErr *= self.dlinear(z,self.unlin_args[Layer]) 
        elif self.unlinearity[Layer] == 1:
            dErr *= self.dsigmoid(z,self.unlin_args[Layer]) 
        elif self.unlinearity[Layer] == 2:
            dErr *= self.dsoftmax(z,self.unlin_args[Layer])        
        elif self.unlinearity[Layer] == 3:
            dErr *= self.drelu(z,self.unlin_args[Layer])
        elif self.unlinearity[Layer] == 4:
            dErr *= 1# self.dheaviside(z,self.unlin_args[Layer])    
        elif self.unlinearity[Layer] == 5:
            dErr *= self.dtanh(z,self.unlin_args[Layer])
        return dErr

    def get_weights(self, Layer = 0):
        dat = pd.read_csv(self.title+"_weights"+str(Layer)+".csv")
        dat = dat.values
        return dat                
    
    def safe_weights(self, Layer, weights, new = True):
        if new:
            o=open(self.title+"_weights"+str(Layer)+".csv","w")
            for d in range(int(weights.shape[1]-1)):
                o.write("w"+str(d)+",")
            o.write("w"+str(weights.shape[1]))
            o.write("\n")    
            o.close()
        try:
            if not np.any(np.isnan(weights)):
                d = open(self.title+"_weights"+str(Layer)+".csv","a")
                for  line in weights:
                    iline =0
                    for field in line:
                        d.write(str(float(field)))
                        if iline<int(weights.shape[1]-1):
                           d.write(",") 
                        iline+=1
                    d.write("\n")
                d.close()
            else:
                print("Fehler!")
                sys.exit(0)
        except:
            print("Dateizugriff nicht möglich")
            sys.exit(0)
    
    def safe_bias(self,Layer,bias,new=True):
        dim=self.dLayers[Layer+1]
        rows = 1 
        columns = dim        
        if new:
            o=open(self.title+"_bias"+str(Layer)+".csv","w")
            for d in range(int(columns-1)):
                o.write("b"+str(d)+",")
            o.write("b"+str(columns-1+1))
            o.write("\n")    
            o.close()
            
        try:
            if not np.any(np.isnan(bias)):
                d = open(self.title+"_bias"+str(Layer)+".csv","a")
                for row in range(rows):
                    for column in range(columns):
                        if rows == 1:
                            d.write(str(float(bias[column])))
                        else:
                            d.write(str(float(bias[row,column])))
                        if column < columns-1:
                           d.write(",") 
                    d.write("\n")
                d.close()
            else:
                print("Fehler!")
                sys.exit(0)
        except:
            print("Dateizugriff nicht möglich")
            sys.exit(0)

    def get_bias(self, Layer = 0):
        dat = pd.read_csv(self.title+"_bias"+str(Layer)+".csv")
        dat = dat.values
        rows,columns=dat.shape
        if rows == 1:
            dat= dat[0]
        return dat        
        

# the initialization of the weights and biases can be done seperately with the methods initalize_weights and initialize_biases
# the method initialize will run both of the above methods
    
    def initialize(self):
        self.initialize_bias()
        self.initialize_weights()
    
    def initialize_weights(self):
        for n in range(self.nLayers-1):
            w = np.random.randn(int(self.dLayers[n]),int(self.dLayers[n+1]))*np.sqrt(2/int(self.dLayers[n]))
            self.safe_weights(n,w,new=True)
            
    def initialize_bias(self):
        for Layer in range(self.nLayers-1):
            dim = self.dLayers[Layer+1]
            b= np.random.randn(dim)#.reshape((dim,1))
            self.safe_bias(Layer,b,new=True)
                
    
        