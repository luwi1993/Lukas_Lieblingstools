# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:36:19 2019

@author: Lukas Widowski
"""
import numpy as np 
import matplotlib.pyplot as plt 


def SVD(M= [[1,0,1,1,0,],[1,1,0,1,0,],[0,1,1,2,1],[0,0,0,0,1]]):
    # Singular value decomposition
    anz_Zeilen,anz_Spalten = np.shape(M)
    [U,S,V] = np.linalg.svd(M)
    #U,S,V=sort_for_biggest_eigenvalue(U,S,V)
    return [U,np.diag(S),V[:anz_Zeilen,:]]


def LSA(M,rank=0,do_find_best_rank=False,plot_it=False):
    # Latent Semantic Analysis/Indexing
    [U,S,V] = SVD(M)

    if do_find_best_rank:
        rank=find_best_rank_integral(np.sum(S,axis=0))
        #rank = find_best_rank_threshold(np.sum(S,axis=0),10**(-3))
    elif rank == 0:
        rank = len(S)
    if plot_it:
        fig = plt.figure()
        plt_S=np.sum(S,axis=0)
        plt.plot(plt_S)
        fig.suptitle("best rank = "+str(rank))
    
    U = U[:,:rank]
    S = S[:rank,:rank]
    V = V[:rank,:]
    return U.dot(S).dot(V),U,S,V
   
def sort_for_biggest_eigenvalue(U,S,V):
    print(S)
    a=zip(S,U.T,V)
    b = sorted(a,reverse=True)
    c=zip(*b)
    d=list(c)
    sorted_lists=d
    #sorted_lists =list(zip(*sorted(zip(S,U.T,V),reverse=True)))
    print(sorted_lists)
    return np.array(sorted_lists[1]).T,np.array(sorted_lists[0]),np.array(sorted_lists[2])

def find_best_rank_integral(S):
    Total = sum(S)
    sum_ = 0 
    rank = len(S)
    for index in range(len(S)-1):
        sum_+= (S[index]+S[index+1])/2
        if sum_ >= 0.9*Total :
            rank = index
            break
    return rank

def find_best_rank_threshold(S,Threshold):
    S,P,len_S=Algorithm_1(S)
    return Algorithm_2(S,P,len_S,Threshold)

def Algorithm_1(S):
    Total = sum(S)
    sum_=0
    P=-1
    len_S=len(S)
    for index1 in range(len_S):
        sum_ += S[index1]
        if sum_/Total > 0.5 :
            P = index1
            break
    if P == -1:
        print("error")
    Total = 0 
    
    for index2 in range(len_S-P):
        index2+=P
        Total += S[index2]
        
    for index3 in range(len_S-P):
        index3+=P
        S[index3] = S[index3]/Total
    
    return S[P:],P,len_S

def Algorithm_2(S,P,len_S,Threshold=10**(-3)):
    for index in range(len_S-P):
        index+=P+1
        if (S[index]-S[index-1]) < Threshold:
            break
    return index


def test(M=np.array([[1,0,1,1,0,],[1,1,0,1,0,],[0,1,1,2,1],[0,0,0,1,1]]),rank=2,q = np.array([1,1,2,1])):
  
    print("Matrix")
    print(M)
    print()
    [U,S,V] = SVD(M)
    approx,U,S,V = LSA(M,rank,True,True)
    print("LSA Approximation")
    [print(L) for L in np.round(approx,2)]    
    print()  
    print("Concepts")
    [print(L) for L in np.round(U,2)]  
    print()
    print("Distribution of Concepts")
    [print(L) for L in np.round(S.dot(V),2)]  
    print()
    print("Mapping Query",q)
    print(q.dot(U))

