#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 19:53:51 2018

@author: schmidt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    
from mpl_toolkits.mplot3d import Axes3D   
from sklearn.utils import shuffle  
from sklearn.preprocessing import StandardScaler
from scipy import signal
import csv_Handler
import sys 
#from plotAssistant import plt3d


class NeuralNetwork:
    
    def __init__(self, title = "", dLayers = [], unlinearity = [], unlin_args = [], Cerr = 0,bias = True, isCNN =False,isRNN = False, start_conv=0,nTimesteps=1):
        self.title = title
        self.nInputs = dLayers[0]
        self.nOutputs = dLayers[-1]
        self.nLayers = len(dLayers)
        self.nWeightsets = self.nLayers-1
        self.Cerr = Cerr
        self.bias = bias

    
        #the unlininearitys are chosen by a vector which defines the Unlinearitys in each Layer
        # 0: no unlinearity              
        # 1: sigmoid      
        # 2: softmax 
        # 3: relu
        # 4: heaviside
        # 5: tanh 
        
        if np.size(unlinearity) == self.nWeightsets:
            self.unlinearity = unlinearity 
        else:
            # if no unlinearitys are chosen by the user sigmod is choosen for every hidden Layer and softmax for the output Layer
            self.unlinearity = 1*np.ones(self.nWeightsets)
            self.unlinearity[-1] = 2 
            print("No unlinearities chosen! -> default Setup: hidden Layers: sigmoid, output Layer: softmax")
        
        # every unlinearity can be modified by giving one argument. 
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
        
        self.isCNN=isCNN            
        self.start_conv = 0
        self.dLayers = dLayers 
        
        # recognition of CNN by initialisation variable dLayers: 
        # is CNN ->   changeing Inputs and start_conv is Transition between CNN and fully connected ANN 
        # is not CNN -> add a bias to the Inputlayer 
        if len(list( i for i in dLayers if type(i)==list)) == 0:
            self.isCNN = False
#            self.dLayers[0]+=1
        else:
            self.isCNN = True 
            self.nInputs = self.nInputs[0]*self.nInputs[1]
            self.start_conv = len(list( i for i in dLayers if type(i)==list))-1
        
        # for a Recursiv Neural Network the Number of timesteps through the recurrent unit has to be specified
        self.isRNN=isRNN
        self.nTimesteps = nTimesteps
        
        # the Width of the computation is an extra parameter which is dependent on the type of model 
        if self.isRNN:        
            self.Width=self.nTimesteps+1
        else:
            self.Width=self.nWeightsets

        
        
        
#__________weights and bias GET and SET _____________:
# the Methods get/set_weights/bias handle csv Data needed to save and return the Weights and Biases of the Neural Network
# the weights of each Layer can be handes seperately and is saved in a seperate csv 
# 
    def get_weights(self, Layer = 0):
        if self.isRNN and Layer>=1 and Layer <= self.Width:
            if Layer == self.Width:
                Layer = 2
            else:
                Layer = 1 
            
        dat = pd.read_csv(self.title+"_weights"+str(Layer)+".csv")
        dat = dat.values
        return dat                
    
    def safe_weights(self, Layer, weights, new = True):
        if new:
            o=open(self.title+"_weights"+str(Layer)+".csv","w")
            for d in range(int(weights.shape[1]-1)):
                o.write("w"+str(d)+",")
            o.write("w"+str(d+1))
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
        if np.size(dim)!=1:
            rows = dim[0]
            columns = dim[1]
        else:
            rows = 1 
            columns = dim
        
        if Layer == self.start_conv-1:
            columns = rows*columns
            rows = 1
        
        if new:
            o=open(self.title+"_bias"+str(Layer)+".csv","w")
            for d in range(int(columns-1)):
                o.write("b"+str(d)+",")
            o.write("b"+str(d+1))
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
        if self.isRNN and Layer>=1 and Layer <= self.Width:
            if Layer == self.Width:
                Layer = 2
            else:
                Layer = 1 
        dat = pd.read_csv(self.title+"_bias"+str(Layer)+".csv")
        dat = dat.values
        rows,columns=dat.shape
        if rows == 1:
            dat= dat[0]
        return dat        
        
# To be able to save the current weights and biases of the model and restore them after wards there are the methods safe_backp and restore_backup 
    def safe_backup(self):
        for Layer in range(self.nWeightsets):
            weights = self.get_weights(Layer)
            o=open(self.title+"_weights"+str(Layer)+"_Backup"+".csv","w")
            for d in range(int(weights.shape[1]-1)):
                o.write("w"+str(d)+",")
            o.write("w"+str(d+1))
            o.write("\n")    
            o.close()
            try:
                if not np.any(np.isnan(weights)):
                    d = open(self.title+"_weights"+str(Layer)+"_Backup"+".csv","a")
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
                
    def restore_backup(self):
        for Layer in range(self.nWeightsets):
            dat = pd.read_csv(self.title+"_weights"+str(Layer)+"_Backup"+".csv")
            weights = dat.values
            self.safe_weights(Layer,weights)

# the initialization of the weights and biases can be done seperately with the methods initalize_weights and initialize_biases
# the method initialize will run both of the above methods
    def initialize(self):
        self.initialize_bias()
        self.initialize_weights()
    
    def initialize_weights(self):
        for n in range(self.nWeightsets):
            if n<self.start_conv:
                x=self.dLayers[n]
                h=self.dLayers[n+1]
                #w = np.random.randn(int(x[0])-int(h[0])+1,int(x[1])-int(h[1])+1)
                w = np.random.randn(int(x[0])-int(h[0])+1,int(x[1])-int(h[1])+1)     
            elif n>0 and n==self.start_conv:
                h=self.dLayers[n]
                w = np.random.randn(int(self.dLayers[n+1]),int(h[0]*h[1]))*np.sqrt(2/int(h[0]*h[1]))
            else:
                w = np.random.randn(int(self.dLayers[n+1]),int(self.dLayers[n]))*np.sqrt(2/int(self.dLayers[n]))
            self.safe_weights(n,w)
            
    def initialize_bias(self):
        for Layer in range(self.nWeightsets):
            if Layer==self.start_conv-1:
                dim = self.dLayers[Layer+1] 
                b= np.random.randn(dim[0]*dim[1])
                self.safe_bias(Layer,b)
            if Layer<self.start_conv-1:
                dim = self.dLayers[Layer+1]
                rows = dim[0]
                columns = dim[1]
                b= np.random.randn(rows,columns)
                self.safe_bias(Layer,b)
            if Layer>self.start_conv-1:
                dim = self.dLayers[Layer+1]
                b= np.random.randn(dim)
                self.safe_bias(Layer,b)
                
# the training or test data can be safed in a csv, which is connected to the model by name identificator.
# either a single vector or a List of vectors (Matrix) can be saved.
# this von be done with the methods safe_date_V (Vector) or safa_data_M (Matrix) 
    def safe_data_M(self, X, T,new = False,Test=False):
        if new:
            if Test:
                o=open(self.title+"_Test_Data"+".csv","w")
            else:
                o=open(self.title+"_Train_Data"+".csv","w")
            for a in range(self.nInputs):
                o.write("X"+str(a)+",")
            for b in range(self.nOutputs-1):
                o.write("T"+str(b)+",")
            o.write("T"+str(self.nOutputs-1))
            o.write("\n")    
            o.close()
        
        try:
            D,N=np.shape(X)
            if self.isCNN:
                X.resize((D*N,))
                self.safe_data_V(X,T,Test=Test)
            else:
                for n in range(N):
                    self.safe_data_V(X[:,n],T[:,n],Test=Test)
        except:
            self.safe_data_V(X,T,Test=Test)
    
    def safe_data_V(self,X,T,new = False,Test=False):
        if new:
            if Test:
                o=open(self.title+"_Test_Data"+".csv","w")
            else:
                o=open(self.title+"_Train_Data"+".csv","w")
            for a in range(self.nInputs):
                o.write("X"+str(a)+",")
            for b in range(self.nOutputs-1):
                o.write("T"+str(b)+",")
            o.write("T"+str(self.nOutputs-1))
            o.write("\n")    
            o.close()
            
        try:
            if Test:
                d=open(self.title+"_Test_Data"+".csv","a")
            else:
                d=open(self.title+"_Train_Data"+".csv","a")
            for x in X:
                d.write(str(x)+",")
            for n in range(len(T)-1):
                d.write(str(T[n])+",")
            d.write(str(T[self.nOutputs-1]))
            d.write("\n")
            d.close()
        except: 
            print("Dateizugriff nicht möglich")
            sys.exit(0)

# the methods get_data returns the Training or Test data which had been saved before       
    def get_data(self,Test=False):
        if Test:
            dat = pd.read_csv(self.title+"_Test_Data"+".csv")
        else:
            dat = pd.read_csv(self.title+"_Train_Data"+".csv")
        dat = dat.values
        X = dat[:,:self.nInputs]
        T = dat[:,self.nInputs:self.nInputs+self.nOutputs]
        N,D=X.shape
        if self.isCNN:
            X=X.T.reshape((self.dLayers[0][0],self.dLayers[0][1],N))
        
        return X.T,T.T

# the method clean_data sorts out items in the Data which appear more than once in exactly the same way 
# the number of items before and after will be printed out
#with the parameter safe you can choose if the List chould be modified or just checked
    def clean_data(self,safe = True,Test=False):
        if Test:
            A = pd.read_csv(self.title+"_Test_Data"+".csv").values
        else:
            A = pd.read_csv(self.title+"_Train_Data"+".csv").values
        print("Anz Datensets vorher:",str(len(A)))
        for n in range(len(A)):
            if n > len(A)-1:
                break
            arg = np.delete(np.argwhere(np.abs(A-np.outer(np.ones(len(A)),A[n])).dot(np.ones(self.nInputs+self.nOutputs ))==0),0)

            for i in range(len(arg)):
             #   print(arg)
                A=np.delete(A,arg[len(arg)-1-i],0)
                if(safe==False):
                    print(arg[len(arg)-1-i])            
                
        print("Anz Datensets nachher:",str(len(A)))
        
        if safe:
            if Test:
                o=open(self.title+"_Test_Data"+".csv","w")
            else:
                o=open(self.title+"_Train_Data"+".csv","w")
            for a in range(self.nInputs):
                o.write("X"+str(a)+",")
            for b in range(self.nOutputs-1):
                o.write("T"+str(b)+",")
            o.write("T"+str(b+1))
            o.write("\n")    
            o.close()
            for s in A:
                X = s[:self.nInputs]
                T = s[self.nInputs:self.nInputs+self.nOutputs]
                self.safe_data_M(X,T,Test=Test)

#the method equalize takes training data and target as input and returns modified data and target which is equally distributed for all possible targets. 
# each target appears as often as the lowest appearence of the input target 
    def equalize(self,x,t):
        D,N=np.shape(x)
        K,N_=np.shape(t)
        
        if N != N_:
            raise ValueError("Input und target haben unterschiedliche Anzahl Samples")    
        l=[]
        for i in range(K): 
            l.append(np.sum(t[i]))
        min_count= min(l)
            
        for n in range(K):
            x,t=clean_target(x,t,n,l[n]-min_count)
            
        return x,t
            
    
    def equalize_data(self,Test=False):
        x,t=self.get_data(Test=Test)
        x,t=self.equalize(x,t)
        self.safe_data_M(x,t,new=True)

# the method get_data_distribution shows the distribution of data along the targets     
    def get_data_distribution(self,Test=False,plot=True):
        x,t=self.get_data(Test=Test)
        D,N=t.shape
        ret=np.zeros(D)
        for i in range(D):
            ret[i]=sum(t[i])
        if plot:
            plt.bar(np.arange(D),ret)
        return ret
    
    def get_data_duplicates(self,Test=False):
        if Test:
            A = pd.read_csv(self.title+"_Test_Data"+".csv").values
        else:
            A = pd.read_csv(self.title+"_Train_Data"+".csv").values
        ret=np.zeros(len(A[0]))
        for n in range(len(A)):
            if n > len(A)-1:
                break
            arg = np.delete(np.argwhere(np.abs(A-np.outer(np.ones(len(A)),A[n])).dot(np.ones(self.nInputs+self.nOutputs ))==0),0)
            if arg.size >0:
                ret=np.vstack((ret,A[arg[0]])) 
        return np.delete(ret,0,0)    
    
    def get_data_Error(self,Test=False):
        X,T = self.get_data(Test=Test)
        return self.Error(X,T)
    
    def get_data_classification_rate(self,Test=False):
        x,t=self.get_data(Test=Test)
        if self.isCNN:
            cl_rate=0
            N=np.shape(x)[0]
            for n in range(N):
                 tt=np.argmax(t[:,n],0)
                 y=np.argmax(self.get_prediction(x[n],self.Width),0)
                 cl_rate+=self.classification_rate(tt,y)
                 print(n/N)
            return cl_rate/N
        else:
            t=np.argmax(t,0)
            y=np.argmax(self.get_prediction(x,self.Width),0)
            return self.classification_rate(t,y)

    def train_data(self, lr=0.0001,nupdates=10,it=10,countdown = True,plot=False,Test=False,dyn_lr=False,l1_w=0,l2_w=0,l1_b=0,l2_b=0):
        x,t = self.get_data(Test=Test)
        if np.size(np.shape(x))==3:
            for n in range(len(x[:,0,0])):
                print("Sample",n,"von",len(x[:,0,0])-1)
                self.fit(x[n],t[:,n],lr,nupdates=nupdates,iterations = it ,countdown = countdown,dyn_lr=dyn_lr,l1_w=l1_w,l2_w=l2_w,l1_b=l1_b,l2_b=l2_b)
        else:
            self.fit(x,t,lr,nupdates=nupdates,iterations = it ,plot=plot,Test=Test,countdown = countdown,dyn_lr=dyn_lr,l1_w=l1_w,l2_w=l2_w,l1_b=l1_b,l2_b=l2_b)
        return 1 
    
    def get_data_pred(self,Test=False,plot= True,dyn_Err = False,argmax = True,c="k"):
        x,t = self.get_data(Test=Test)  
        D,N = np.shape(x)
        pred = self.pred(x,self.nLayers-1)
        if argmax == True:
            pred_max = np.argmax(pred,0)
            t_max = np.argmax(t,0)
        
        if plot == True:
            ax=np.arange(N)
            plt.plot(ax,sum(t*pred,0),label = "Error",c=c)
            if dyn_Err:
                Err = np.zeros(N)
                for n in range(N):
                    Err[n]= self.Error(x[:,n],t[:,n])
                plt.plot(ax,Err)  
        return pred_max,t_max,pred_max==t_max

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
        if self.isRNN:
            if Layer>=self.Width+1:
                Layer=3
            elif Layer == 1:
                pass
            else:
                Layer = 2
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
        if self.isRNN:
            if Layer>=self.Width+1:
                Layer=2
            elif Layer == 0:
                pass 
            else:
                Layer = 1  
        elif self.unlinearity[Layer] == 0:
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

#__________Predictions and Gradient ____________:  
            
    def Error(self,x,t): 
        y = self.get_prediction(x)
        if self.dLayers[-1]>1:
            return -np.mean(t*np.log(y))
        else:
            return sum((t-y)**2)/len(y)
            
    
    def classification_rate(self,t,pred):
        return np.mean(t==pred)
    
    def get_prediction(self,x,Layer=0):
        Width = self.Width
        if self.isRNN:
            Width +=1
        return self.pred(x,Width)
    
    def dim(self,x,components=False):
        x_dim=np.size(x.shape)
        N,K,D=0,0,0
        if components:            
            if x_dim==1:
                N=1
                D=len(x)
            elif x_dim==2:
                if self.isCNN or self.isRNN:
                    D,K = np.shape(x)
                else:
                    D,N = np.shape(x)
            elif x_dim==3:
                N,K,D= np.shape(x)
            return N,D,K
        else: 
            return x_dim
        
    def convolve3d(self,x,z,mode="valid",parallel=False):
        if parallel:
            y=[]
            y.append(signal.convolve2d(x[0],z[0],mode))
        else:
            y=signal.convolve2d(x[0],z[0],mode)
        for i in range(len(z)-1):
            if parallel:
                y.append(signal.convolve2d(x[i+1],z[i+1],mode))
            else:
                y+=signal.convolve2d(x[i+1],z[i+1],mode)
        if parallel:
            y=np.array(y)
        return y
        
    def Range_check(self,K,Layer,dErr = False):
        if self.isRNN:        
            if K>1 and self.isCNN == False:
                self.nTimesteps = K-1
            self.Width=self.nTimesteps+1
            Break = self.Width +1
        else:
            self.Width=self.nWeightsets
            Break = self.Width
        
        if dErr:
            Break-=1
            
        if Layer > Break:
            print("Geltungsbereich überschritten: Layer = "+str(Layer)+" -> Max:"+str(Break)+"\n\n")
            sys.exit(0) 
    
    def pred(self,x,Layer): 
        N,D,K = self.dim(x,True)
        self.Range_check(K,Layer)
        
        if Layer==0:
            if self.isRNN:
                return x[:,0]
            else:
                return x
        elif Layer<=self.start_conv:
            w = self.get_weights(Layer-1)
            if N>1:
                y=self.convolve3d(self.pred(x,Layer-1),w,mode="valid",parallel=True)
            else:    
                y=signal.convolve2d(self.pred(x,Layer-1),w,mode="valid")             
            if Layer==self.start_conv:
                (a,b)=np.shape(y)
                y.resize((a*b,))
        else:
            w = self.get_weights(Layer-1)
            y = w.dot(self.pred(x,Layer-1)) 
        
        if self.isRNN and Layer < self.Width-1:
            u=self.get_weights(0)
            y+=u.dot(x[:,Layer])
            
        if self.bias:
            b=self.get_bias(Layer-1)
            if np.shape(y)!=np.shape(b):
                y+= np.outer(b,np.ones(y[0].size))
            else:
                y+= b
        
        return self.get_unlinearity(y,Layer)

    def get_dErr(self,x,t,Layer):
        N,D,K = self.dim(x,True)

        self.Range_check(K,Layer,True)
            
        y = self.get_prediction(x,self.Width) 
        z = self.pred(x,Layer+1)        
        
        
        if Layer == self.start_conv-1:
            z.resize((self.dLayers[Layer+1][0],self.dLayers[Layer+1][1]))
            
        if Layer >=self.Width:
            if not self.isRNN:
                print("no valid case in get_dErr()")   
                sys.exit(0)
            else:
                return -(t-y*sum(t,0))
            
        if Layer == self.Width - 1 and not self.isRNN:
            return -(t-y*sum(t,0)) 
             
        elif Layer >= self.start_conv-1:
            dErr = self.get_weights(Layer+1).T.dot(self.get_dErr(x,t,Layer+1))
            if Layer == self.start_conv-1:
                dErr.resize((self.dLayers[Layer+1][0],self.dLayers[Layer+1][1]))
        
        elif Layer < self.start_conv-1:
            dErr = signal.convolve2d(self.get_weights(Layer+1),self.get_dErr(x,t,Layer+1))
        
        else: 
            print("no valid case in get_dErr()")    
            
        return self.get_dunlinearity(dErr,z,Layer)

    def fit(self,x,t,learning_rate=0.001,iterations=1,nupdates=1,l1_w=0,l2_w=0,l1_b=0,l2_b=0,countdown = False, dyn_lr = False, min_lr = 10**(-10),plot=False,Test=False):
        N,D,K = self.dim(x,True)
        
        if not self.isCNN and not self.isRNN and N>1:
            multiSample = True
        else:
            multiSample = False
        
        self.Range_check(K,0)   
        if self.isRNN:        
            Width = self.Width+1
        else:
            Width = self.Width
                   
        bias_Gradient = 0 
        weights_Gradient = 0
        RNN_In_weights_Gradient = 0 
        pre_lr = learning_rate
        preerr=self.Error(x,t) 
        if plot:
            x_plot,t_plot = self.get_data(Test=Test)
        
        for update in range(nupdates):
            for Layer in range(Width-1,-1,-1): 
                
                if dyn_lr:
                    learning_rate=pre_lr
    
                for i in range(iterations):   
                    miderr = self.Error(x,t)                  
                    # Derivation of the Gradient 
                    if N == 1: 
                        bias_Gradient = self.get_dErr(x,t,Layer)
                        weights_Gradient = np.outer(bias_Gradient,self.pred(x,Layer))
                    
                    elif self.isRNN:
                        bias_Gradient = self.get_dErr(x,t,Layer)
                        if Layer == self.Width:
                            RNN_Out_weights_Gradient= np.outer(bias_Gradient,self.pred(x,Layer))
                        else:
                            if Layer == 0: 
                                pred = np.zeros(self.dLayers[1])
                            else:
                                pred = self.pred(x,Layer)
                            weights_Gradient += np.outer(bias_Gradient,pred)
                            RNN_In_weights_Gradient += np.outer(bias_Gradient,x[:,Layer])
                    
                    elif multiSample:  
                        bias_Gradient= sum(self.get_dErr(x,t,Layer),1)
                        weights_Gradient= self.get_dErr(x,t,Layer).dot(self.pred(x,Layer).T)
                        
                    elif self.isCNN: 
                        if Layer >= self.start_conv:
                            bias_Gradient = self.get_dErr(x,t,Layer)
                            weights_Gradient = np.outer(bias_Gradient,self.pred(x,Layer))
                        if Layer < self.start_conv:
                            bias_Gradient = self.get_dErr(x,t,Layer)
                            weights_Gradient=signal.convolve2d(self.pred(x,Layer),bias_Gradient,"valid")  
                    else:
                        change_in_Error= self.Error(x,t)-miderr
                        print("invalid case in fit()")
                        print("Error while: update",str(update),"von",str(nupdates),"Layer",str(self.Width-Layer),"von",str(self.Width),"Iteration",str(i+1),"von",str(iterations),"Error=",str(self.Error(x,t)),"classificationrate=",str(self.classification_rate(np.argmax(t,0),np.argmax(self.get_prediction(x,self.Width),0))),"Learningrate",str(learning_rate),"Change in Error=", str(change_in_Error))

                    # Gradient Descent 
                    if self.isRNN==False or Layer == 0:
                        if self.isRNN==True:
                            Layer = 1

                        w = self.get_weights(Layer)
                        w -=learning_rate*weights_Gradient+l1_w*np.sign(w)+l2_w*w
                        
                        # Saving new weights 
                        pre_w = self.get_weights(Layer)                   
                        self.safe_weights(Layer,w)
                        
                        # same for the bias (if needed)
                        if self.bias == True and not self.isRNN:
                            b = self.get_bias(Layer)
                            if Layer == self.start_conv-1:
                                bias_Gradient.resize((np.shape(b)))
                            b -= learning_rate*bias_Gradient+l1_b*np.sign(b)+l2_b*b
                            self.safe_bias(Layer,b)
                        
                        if self.isRNN:
                             u = self.get_weights(0)
                             u -=learning_rate*RNN_In_weights_Gradient+l1_w*np.sign(u)+l2_w*u
                             self.safe_weights(0,u)
                             v = self.get_weights(self.Width)
                             v -=learning_rate*RNN_Out_weights_Gradient+l1_w*np.sign(v)+l2_w*v             
                             self.safe_weights(2,v)
                             if self.isRNN:
                                 b = self.get_bias(Layer)
                                 b -= learning_rate*bias_Gradient+l1_b*np.sign(b)+l2_b*b
                                 self.safe_bias(1,b)

                            
                        
                        # dynamic learning_rate Kontrol            
                        change_in_Error= self.Error(x,t)-miderr
                        if dyn_lr:                            
                            while change_in_Error >= 0:
                                learning_rate *= 0.8
                                w = pre_w - learning_rate*weights_Gradient-l1_w*np.sign(w)-l2_w*w
                                self.safe_weights(Layer,w)
                                change_in_Error= self.Error(x,t)-miderr
                                if change_in_Error<0:
                                    break
                                if learning_rate< min_lr:
                                    print("No Progress: Local Minima -> Gradent Backstep:","Error=",str(self.Error(x,t)),"classificationrate=",str(self.classification_rate(np.argmax(t,0),np.argmax(self.get_prediction(x,self.Width),0))),"Learningrate",str(learning_rate),"Change in Error=", str(change_in_Error))
                                    pre_lr*=0.5
                                    if pre_lr < min_lr:
                                        print("\n \n Error! -> Sys Exit: fix Hyperparameters or initialize weights:","Error=",str(self.Error(x,t)),"classificationrate=",str(self.classification_rate(np.argmax(t,0),np.argmax(self.get_prediction(x,self.Width),0))),"Learningrate",str(learning_rate),"Change in Error=", str(change_in_Error))
                                        sys.exit(0)
                                    w = pre_w + pre_lr*weights_Gradient
                                    learning_rate=pre_lr
                                    self.safe_weights(Layer,w)
                                    break
                        
                        # printing out the Progress 
                        if countdown == True and i%10==0:   
                            change_in_Error= self.Error(x,t)-miderr
                            print("Finished: update",str(update),"von",str(nupdates),"Layer",str(self.Width-Layer),"von",str(self.Width),"Iteration",str(i+1),"von",str(iterations),"Error=",str(self.Error(x,t)),"classificationrate=",str(self.classification_rate(np.argmax(t,0),np.argmax(self.get_prediction(x,self.Width),0))),"Learningrate",str(learning_rate),"Change in Error=", str(change_in_Error))   
                        
            if plot and update%20==0:
                D,N = np.shape(x_plot)
                pred = self.pred(x_plot,self.nLayers-1)
                ax=np.arange(N)
                plt.plot(ax,sum(t_plot*pred,0),label = 'update nr.'+str(update)+" Error",alpha=0.5)
                plt.legend()
                            
                            
                            
        if countdown == True:
            print("TOTAL CHANGE IN ERROR:"+str(self.Error(x,t)-preerr))

    def RNN_fwd(self,X):
        preY = np.zeros(self.nOutputs)
        for x in X.T:
            In = np.hstack((x,preY))
            print(In)
            preY = self.pred(In,self.nLayers-1)
        return preY
    
    def delayed_RNN_fwd(self,x,nSteps=1):
        preY1=x
        preY2 = np.zeros(self.nOutputs)
        for n in range(nSteps):
            In = np.hstack((preY1,preY2))
            preY2=preY1
            preY1= self.pred(In,self.nLayers-1)
            print(In,preY1)
        return preY1
    
#    def dynamic_learning_rate(self,Layer,x,t,learning_rate, pre_err, pre_w ,pre_lr, Gradient ,l1=0,l2=0):
#        weights = self.get_weights(Layer)
#        change_in_Error= self.Error(x,t)-pre_err
#        while change_in_Error >= 0:
#            learning_rate *= 0.8
#            weights = pre_w - learning_rate*Gradient+l1*np.sign(weights)+l2*weights
#            self.safe_weights(Layer,weights)
#            change_in_Error= self.Error(x,t)-pre_err
#            if change_in_Error<0:
#                break
#            if learning_rate<10**(-10):
#                print("Error! -> No Progress: fix learning rate or initialize weights-> Gradent Backstep","Error=",str(self.Error(x,t)),"classificationrate=",str(self.classification_rate(np.argmax(t,0),np.argmax(self.get_prediction(x,self.nLayers-1),0))),"Learningrate",str(learning_rate),"Change in Error=", str(change_in_Error))
#                pre_lr*=0.7
#                w = pre_w + pre_lr*Gradient
#                learning_rate=pre_lr
#                self.safe_weights(Layer,w)
#                break
#        return learning_rate,weights
#        

#______________Testmethods______________:
 
def num_to_vec(t):
    M = np.zeros((max(t)+1,len(t)))
    for n in range(len(t)):
        M[t[n]][n]=1
    return M 

def preprocess_ecommerce():
    dat = pd.read_csv("ecommerce_data.csv")
    val = dat.values
    val=shuffle(val)
    X=val[:,:-1]
    Y=val[:,-1]
    N = len(Y)
    Y2 = np.zeros((N,int(max(Y))+1))
    for n in range(N):
        Y2[n][int(Y[n])]=1
    Y2 = Y2.T
    X2 = np.zeros((N,8))
    X2[:,0]= X[:,0]
    X2[:,1]= (X[:,1]-X[:,1].mean())/X[:,1].std()
    X2[:,2]= (X[:,2]-X[:,2].mean())/X[:,2].std()
    X2[:,3]= X[:,3]
    for n in range(N):
        X2[n][4+int(X[n][4])]=1
    X2=X2.T
    Xtrain = X2[:,:-100]
    Ytrain = Y2[:,:-100]
    Xtest = X2[:,-100:]
    Ytest = Y2[:,-100:]
    return Xtrain,Ytrain,Xtest,Ytest



def rolfs_datenarray(Spalte):
    dat = pd.read_csv("Mappe1.csv")
    dat = dat.values
    return dat[:,Spalte] 
        
def PCA(x,reduction = 0):
    val,vec = np.linalg.eig(np.cov(x))
    pairs  = [(val[i],vec[i]) for i in range(np.shape(x)[0])]
    pairs.sort(reverse=True)
    M = pairs[0][1][:,np.newaxis]
    for n in range(np.shape(x)[0]-1-reduction):
        M = np.hstack((M,pairs[n+1][1][:,np.newaxis]))
    return x.T.dot(M).T

def LDA(x,t,reduction=0):
    
    d = x.shape[0]
    mean_vecs = []
    for i in range(t.shape[0]):
        mean_vecs.append(np.mean(x.T[np.argmax(t,0)==i],axis=0))
    S_W = np.zeros((d,d))
    for label,mv in zip(range(0,4),mean_vecs):
        cx=x.T[np.argmax(t,0)==label].T
        class_scatter = np.cov(cx)
        S_W += class_scatter 
    S_B = np.zeros((d,d))
    mean_overall = np.mean(x.T,0)
    for i,mean_vec in enumerate(mean_vecs):
        n=x.T[np.argmax(t,0)==i].shape[0]
        mean_vec,mean_overall=mean_vec.reshape(d,1),mean_overall.reshape(d,1)
        S_B += n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
    eigen_vals,eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs,key = lambda k:k[0],reverse=True)
    
    M = eigen_pairs[0][1][:,np.newaxis]
    for n in range(np.shape(x)[0]-1-reduction):
        M = np.hstack((M,eigen_pairs[n+1][1][:,np.newaxis]))
    
    return x.T.dot(M).T
    
def std_scale(x):
    sc = StandardScaler()
    x= sc.fit_transform(x.T)
    x=x.T
    return x

def random_cloud(d=3,n=10):
    r=np.array([np.random.randn((d)) for n in range(n)]).T
    r = (r-np.mean(r))/np.std(r)
    return r

def temp(dat,boollist):
    for i in range(len(boollist)): 
        if not boollist[len(boollist)-1-i]:
            dat=np.delete(dat,len(boollist)-1-i,axis=0)
    return dat 

def clean(dat):
    zeilen,spalten = dat.shape        
    for s in range(spalten):
        for z in range(zeilen):
            if not dat[z,s]>0:
                dat[z,s]=0
    return dat 
            


def mk_t(dim=2,n1=0,nSamples=1):
    t=np.zeros((dim,nSamples))
    t[n1]=1
    if nSamples == 1:
        t=t.flatten()
    return t


def clean_target(x,t,target_index,clean_n=0,clean_all=False,axis=1):
    n = 0
    if clean_all:
        clean_n = len(np.argwhere(t[target_index]==1))
    for i in np.argwhere(t[target_index]==1)[::-1]: 
        if n>=clean_n:
            break
        n+=1
                     
        t=np.delete(t,i[0],axis = 1)
        x=np.delete(x,i[0],axis = 1)        
       
    return x,t
    
def equalize_data(x,t):
    D,N=np.shape(x)
    K,N_=np.shape(t)
    
    if N != N_:
        raise ValueError("Input und target haben unterschiedliche Anzahl Samples")    
    l=[]
    for i in range(9): 
        l.append(np.sum(t[i]))
    min_count= min(l)
        
    for n in range(K):
        x,t=clean_target(x,t,n,l[n]-min_count)
        
    return x,t
            
        
        
            

            