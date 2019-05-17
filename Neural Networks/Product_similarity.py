# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:17:00 2019

@author: Lukas Widowski
"""

import pandas as pd 
import numpy as np
import Data_Preprocessor
import matplotlib.pyplot as plt 
import LSA
import LSA_Test
import re
import datetime
import time
from operator import itemgetter
from IPython import get_ipython
from NeuralNetwork import NeuralNetwork
from sklearn.cluster import KMeans
plt.interactive(False)
get_ipython().run_line_magic('matplotlib', 'qt')

def get_data():
    data = pd.read_excel("Beispiel-Superstore.xls",parse_dates=["Bestelldatum"]).values
    selected_data = np.zeros(len(data))
    index_list =[6,15,2]#[6,15,17,2]#[1,8,9,10,17]
    for index in index_list:
        selected_data=np.vstack((selected_data,data[:,index]))
    selected_data=np.array(selected_data).T[:,1:]
    return selected_data    


def get_metadata():
    data = pd.read_excel("Beispiel-Superstore.xls",parse_dates=["Bestelldatum"]).values
    selected_data = np.zeros(len(data))
    index_list =[6,15,17,18,19]
    for index in index_list:
        selected_data=np.vstack((selected_data,data[:,index]))
    selected_data=np.array(selected_data).T[:,1:]
    
    preprocessed_selected_data=Data_Preprocessor.string_to_unique_int(selected_data[:,0])
    preprocessed_selected_data=np.vstack((preprocessed_selected_data,Data_Preprocessor.string_to_unique_int(selected_data[:,1])))
    preprocessed_selected_data = Data_Preprocessor.unobject(preprocessed_selected_data)
    preprocessed_selected_data = np.vstack((preprocessed_selected_data,selected_data[:,2]))  
    preprocessed_selected_data = np.vstack((preprocessed_selected_data,selected_data[:,3]))  
    preprocessed_selected_data = np.vstack((preprocessed_selected_data,selected_data[:,4]))  
    
    sorted_preprocessed_selected_data=np.array(sorted(preprocessed_selected_data.T, key=itemgetter(1),reverse=False))
    n_Produkte = int(max(sorted_preprocessed_selected_data[:,1]))+1
    Produkte = [[] for i in range(n_Produkte)]
    
    for dat in sorted_preprocessed_selected_data:
        Produkte[int(dat[1])].append(dat)

    Avg_Menge = [[] for i in range(n_Produkte)]
    Avg_Rabatt = [[] for i in range(n_Produkte)]
    Avg_Gewinn = [[] for i in range(n_Produkte)]
    Avg_Gewinn_pro_stk = [[] for i in range(n_Produkte)]
    
    for i in range(n_Produkte):
        Avg_Menge[i] =np.mean(np.array(Produkte[i])[:,2])
        Avg_Rabatt[i]=np.mean(np.array(Produkte[i])[:,3])
        Avg_Gewinn[i]=np.mean(np.array(Produkte[i])[:,4])
        Avg_Gewinn_pro_stk[i] = np.mean(np.array([Produkte[i][n][4]/Produkte[i][n][2] for n in range(len(Produkte[i]))]))
       
    return Avg_Menge,Avg_Rabatt,Avg_Gewinn,Avg_Gewinn_pro_stk
    
def preprocess_data(data):
    preprocessed_data=Data_Preprocessor.string_to_unique_int(data[:,0])
    preprocessed_data=np.vstack((preprocessed_data,Data_Preprocessor.string_to_unique_int(data[:,1])))
    preprocessed_data = Data_Preprocessor.unobject(preprocessed_data)
    preprocessed_data = np.vstack((preprocessed_data,data[:,-1]))  
    return preprocessed_data.T

def get_lookups_and_appearance_Matrix(preprocessed_data,selected_data,data):
    n_Produkte = int(np.max(preprocessed_data[:,1]))+1
    n_Kunden = int(np.max(preprocessed_data[:,0]))+1
    K_lookup = sorted(zip(preprocessed_data[:,0],data[:,0]))
    P_lookup = sorted(zip(preprocessed_data[:,1],data[:,1]))
    # lookup tables halten die namen der Produkte bzw Kunden an den indexen welche sie im algorithmus repräsentieren

    Produkt_lookup = []
    Kunden_lookup = [] 
    P_seen = []
    K_seen = []    
    for i in range(len(K_lookup)):
        if not K_lookup[i][0] in K_seen:
            Kunden_lookup.append(K_lookup[i][1])
            K_seen.append(K_lookup[i][0])
        if not P_lookup[i][0] in P_seen:
            Produkt_lookup.append(P_lookup[i][1])
            P_seen.append(P_lookup[i][0])
        
    # die Kunden_Producte_Matrix enthält die information welcer Kunde welches Product gekauft hat,
    # information über die menge der stellten produkte oder die anzahl der bestellungen wurden heraus gelassen
    Kunden_Produkte_Matrix = np.zeros((n_Kunden,n_Produkte))
    for i in selected_data:
        Kunden_Produkte_Matrix[int(i[0]),int(i[1])]=1
    
    return Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup
    
    
def get_matricies(Kunden_Produkte_Matrix,show_progress=False):
    (n_Kunden,n_Produkte) = np.shape(Kunden_Produkte_Matrix)
    if_occurence = np.zeros((n_Produkte,n_Produkte))
    cooccurence  = np.zeros((n_Produkte,n_Produkte))
    P = np.zeros(n_Produkte)
    #cooccurence[row,col]=max(0,log(P(row,col)/(P(row)*P(col))))
    for row in range(n_Produkte):
        P[row]=sum(Kunden_Produkte_Matrix[:,row])/n_Kunden
        if show_progress:
            print(row/n_Produkte)
        for col in range(row+1):
#            p_row=sum(Kunden_Produkte_Matrix[:,row])/n_Kunden
#            p_col=sum(Kunden_Produkte_Matrix[:,col])/n_Kunden
            p_row = P[row]
            p_col= P[col]
            p_row_and_col = Kunden_Produkte_Matrix[:,row].dot(Kunden_Produkte_Matrix[:,col])/n_Kunden
            if p_row_and_col == 0:
                cooccurence[row,col]= 0    #hier kann ein penalty wert eingesetzt werden default ist  0
            else:
                cooccurence[row,col]=max(0,np.log(p_row_and_col/max(10**(-10),(p_row*p_col))))  
                
            cooccurence[col,row]=cooccurence[row,col]
            #if_occurence[row,col]=P(row|col)
            if_occurence[row,col]=p_row_and_col/p_col if p_col != 0 else 0
            if_occurence[col,row]=p_row_and_col/p_row if p_row != 0 else 0
    occurence = P
    return if_occurence,cooccurence,occurence

def get_suggestion_for_Product(product_index,if_occurence,Produkt_lookup):
    index= sorted(zip(if_occurence[:,product_index],np.arange(len(Produkt_lookup))))[1][1]
    print(np.round(if_occurence[index,product_index]*100,decimals=1),"% der leute die produkt ",Produkt_lookup[product_index]," gekauft haben, haben ebenfalls das Produkt ",Produkt_lookup[index]," gekauft")
    return index,if_occurence[index,product_index]

def get_similarity(product_a_index,product_b_index,cooccurence):
    prod_a_vec=cooccurence[product_a_index]
    prod_b_vec=cooccurence[product_b_index]
    corr_a_b = prod_a_vec.dot(prod_b_vec)
    dist_a = np.sqrt(prod_a_vec.dot(prod_a_vec))
    dist_b = np.sqrt(prod_b_vec.dot(prod_b_vec))
    return  2*(np.arccos(-corr_a_b/max(10**(-5),(dist_a*dist_b)))/np.pi-0.5)

def show_most_similar(product_index,cooccurence,Produkt_lookup=[],print_n=0,word=0):
    if word!=0:
        for index in range(len(Produkt_lookup)):
            if Produkt_lookup[index] == word:
                product_index  = index
                break
    n_Produkte = len(cooccurence)
    similarities=np.array([get_similarity(product_index,n,cooccurence=cooccurence) for n in range(n_Produkte)])
    plt.plot(np.arange(n_Produkte),similarities)
    ret = sorted(zip(similarities,np.arange(n_Produkte)),reverse=False)
    if print_n != 0:
        print("Basisproduct:\t"+Produkt_lookup[product_index])
        for i in range(print_n):
            print(ret[i][0],Produkt_lookup[int(ret[i][1])],int(ret[i][1]))
    return ret 

def get_co_occurence_matrix(show_progress=False):
    data=get_data()
    preprocessed_data=preprocess_data(data)
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)
    #Kunden_Produkte_Matrix=Kunden_Produkte_Matrix*2-1
    if_occurence,co_occurence,occurence = get_matricies(Kunden_Produkte_Matrix,show_progress=show_progress)
    return if_occurence,co_occurence,occurence


def show_word_synonyms(synonyms,Produkt_lookup,product_index=0,n_synonyms=5,word=0,reverse=True):
    if word!=0:
        for index in range(len(Produkt_lookup)):
            if Produkt_lookup[index] == word:
                product_index  = index
                break
        #product_index = np.argwhere(Produkt_lookup==word)
    row = product_index    
    sorted_row=sorted(zip(synonyms[row],np.arange(len(synonyms))),reverse=reverse)
    print("Basiswort: "+Produkt_lookup[row]+"\tIndex: "+str(row))
    for top in range(n_synonyms):
        print(sorted_row[top][0],Produkt_lookup[sorted_row[top][1]])


def show_n_synonyms(n_synonyms,synonyms,Produkt_lookup,reverse=True):
    for row in range(len(synonyms)):
        sorted_row=sorted(zip(synonyms[row],np.arange(len(synonyms))),reverse=reverse)
        print("Basiswort:",Produkt_lookup[row],row)
        for top in range(n_synonyms):
            print(sorted_row[top][0],Produkt_lookup[sorted_row[top][1]])
        print(20*"--")

def find_pattern(raw_string,Produkt_lookup):
    pattern =re.compile(raw_string)
    findings = []
    for i in range(len(Produkt_lookup)):
        matches = pattern.finditer(Produkt_lookup[i])  
        for match in matches:
            findings.append((match,i))
    return findings

def get_synonyms(rank=0,do_find_best_rank=True):
    data=get_data()
    preprocessed_data=preprocess_data(data)
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)
    synonyms=LSA_Test.get_synonyms(Kunden_Produkte_Matrix.T,rank=rank,do_find_best_rank=do_find_best_rank,plot_it=True)
    return synonyms,Produkt_lookup

def get_LSA_Transform(rank=0,do_find_best_rank=True):
    data=get_data()
    preprocessed_data=preprocess_data(data)
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)
    LSA_approx,U,S,V = LSA.LSA(Kunden_Produkte_Matrix,do_find_best_rank==do_find_best_rank)
    return LSA_approx,Produkt_lookup

def Cluster(X,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    return y_kmeans


    
def get_advanced_synonyms(rank=0,do_find_best_rank=True): 
    data=get_data()
    preprocessed_data=preprocess_data(data)
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)
    LSA_approx,U,S,V = LSA.LSA(Kunden_Produkte_Matrix,do_find_best_rank==do_find_best_rank)
    (n_Kunden,n_Produkte) = np.shape(Kunden_Produkte_Matrix)
    ret = np.zeros((n_Produkte,n_Produkte))
    Dist = np.zeros(n_Produkte)
    for r in range(n_Produkte):
        Dist[r] = np.sqrt(LSA_approx[:,r].dot(LSA_approx[:,r]))
        for c in range(r):
            Corr=LSA_approx[:,r].dot(LSA_approx[:,c])
            Dist_r=Dist[r]
            Dist_c=Dist[c]
            ret[r,c]=Corr/max(10**(-5),(Dist_r*Dist_c))
            ret[c,r]=ret[r,c]
         #   L[r,c]=2*(np.arccos(-Corr/max(10**(-5),(Dist_r*Dist_c)))/np.pi-0.5)
    return ret

def get_Kunden_Produkte_Matrix():
    data=get_data()
    preprocessed_data=preprocess_data(data)
    return get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)


def get_avg_Kunde(prod_index,Kunden_Produkte_Matrix,penalty_val=0):
    indexes = np.argwhere(Kunden_Produkte_Matrix[:,prod_index]==1)    
    prod_avg_Kunde = np.ones(len(Kunden_Produkte_Matrix[0]))*(-penalty_val)
    for i in indexes:
        i=i[0]
        prod_avg_Kunde+=Kunden_Produkte_Matrix[i]/sum(Kunden_Produkte_Matrix[i])
    return prod_avg_Kunde/len(indexes)

def get_avg_Kunde2(prod_index,Kunden_Produkte_Matrix,penalty_val=0,n_Produkte = 0):
    prod= Kunden_Produkte_Matrix[:,prod_index]
    return np.sum(Kunden_Produkte_Matrix*np.outer(np.sum(Kunden_Produkte_Matrix,axis=1)**(-1),np.ones(n_Produkte))*np.outer(prod,np.ones(n_Produkte)),axis=0)/sum(prod)

def get_comparison_Matrix(show_progress=False,penalty_val=0):
    data=get_data()
    preprocessed_data=preprocess_data(data)
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,preprocessed_data,data)
    Kunden_Produkte_Matrix = Kunden_Produkte_Matrix
    (n_Kunden,n_Produkte) = np.shape(Kunden_Produkte_Matrix)
    ret = np.zeros((n_Produkte,n_Produkte))
    start_time=time.time()
    history=[]
    
    avg_Kunden = [0 for i in range(n_Produkte)]
    for r in range(n_Produkte):
        if show_progress: 
            print(r/n_Produkte)
            history=get_runtime_history(r/n_Produkte,history,start_time)      
        avg_Kunden[r]=get_avg_Kunde2(r,Kunden_Produkte_Matrix,penalty_val,n_Produkte)
        for c in range(r+1):       
            ret[r,c] = avg_Kunden[r].dot(avg_Kunden[c])
            ret[c,r] = ret[r,c]
    
    if show_progress:
        plt.plot(np.array(history)[:,0],np.array(history)[:,1])
    return ret

def get_käufe_von_kunde(kunden_index,Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup):
    Kunde = Kunden_Produkte_Matrix[kunden_index]
    indexes = np.argwhere(Kunde == 1)
    print("Kunde",Kunden_lookup[kunden_index],"kauft die folgenden Produkte")
    for index in indexes:
        print(Produkt_lookup[index[0]])
        
def get_runtime_history(progress,history=[],start_time=0):
    history.append([time.time()-start_time,progress])
    return history

def sort_for_Clients():
    data=get_data()
    preprocessed_data=preprocess_data(data)
    sorted_preprocessed_data=np.array(sorted(preprocessed_data, key=itemgetter(2),reverse=False))
    n_Kunden = int(max(sorted_preprocessed_data[:,0]))+1
    Clients = [[] for i in range(n_Kunden)]
    for dat in sorted_preprocessed_data:
        Clients[int(dat[0])].append(dat)
    return Clients


def get_delayed_date(from_date,plus_n_weeks=4):
    delta =  datetime.timedelta(weeks=+plus_n_weeks)
    return from_date + delta

def get_data_in_interval(datas,from_date,to_date):
    data_in_interval = []
    for data in datas:
        if (data[-1]>=from_date) and (data[-1]<=to_date):
            data_in_interval.append(data)
        if data[-1]>=to_date:
            break
    return data_in_interval


def get_time_dependent_co_occurence_matrix():
    data=get_data()
    preprocessed_data=preprocess_data(data)
    sorted_preprocessed_data=np.array(sorted(preprocessed_data, key=itemgetter(2),reverse=False))
    min_date=sorted_preprocessed_data[0,2]
    max_date=sorted_preprocessed_data[-1,2]
    from_date=min_date
    to_date=get_delayed_date(min_date)
    if_occurences_list =[]
    co_occurences_list = []
    while to_date<=max_date:
        selected_data=np.array(get_data_in_interval(sorted_preprocessed_data,from_date,to_date))[:,:-1]
        print(max_date,to_date,len(selected_data))
        if len(selected_data)>0:
            Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,selected_data,data)
            if_occurence,co_occurence,occurence = get_matricies(Kunden_Produkte_Matrix,show_progress=False)
            if_occurences_list.append(if_occurence)
            co_occurences_list.append(co_occurence)
            from_date=to_date
            to_date=get_delayed_date(to_date)
            
    result = np.zeros(np.shape(co_occurences_list[0]))
    for matrix in co_occurences_list:
        result+=matrix
        
    result = result/len(co_occurences_list) 
    return result

def get_monthly_Clients(n_weeks = 4):
    data=get_data()
    preprocessed_data=preprocess_data(data)
    sorted_preprocessed_data=np.array(sorted(preprocessed_data, key=itemgetter(2),reverse=False))
    min_date=sorted_preprocessed_data[0,2]
    max_date=sorted_preprocessed_data[-1,2]
    from_date=min_date
    to_date=get_delayed_date(min_date,n_weeks)
    n_steps = int(round((max_date-min_date).days/7+0.5))
    n_Kunden = int(max(sorted_preprocessed_data[:,0]))+1
    n_Produkte = int(max(sorted_preprocessed_data[:,1]))+1
    ret=np.zeros((n_steps,n_Kunden,n_Produkte))
    i = 0 
    while to_date<=max_date:
        selected_data=np.array(get_data_in_interval(sorted_preprocessed_data,from_date,to_date))[:,:-1]
        Kunden_Produkte_Matrix,Produkt_lookup,Kunden_lookup=get_lookups_and_appearance_Matrix(preprocessed_data,selected_data,data)
        ret[i]=Kunden_Produkte_Matrix
        from_date=to_date
        to_date=get_delayed_date(to_date,n_weeks)
        i+=1
    return ret 

def compare_approaches(calc=True,product_index=823,n_synonyms=10,synonyms=0,if_occurence=0,cooccurence=0,comparison_Matrix_0=0,comparison_Matrix_01=0,comparison_Matrix_001=0,Produkt_lookup=0):
    if calc:
        synonyms,Produkt_lookup=get_synonyms()
        if_occurence,cooccurence,occurence=get_co_occurence_matrix(True)
        comparison_Matrix_001 = get_comparison_Matrix(True,0.000000001)
        comparison_Matrix_01 = get_comparison_Matrix(True,0.1)
        comparison_Matrix_0 = get_comparison_Matrix(True,0)
        
    print(100*"-")
    print("LSA\n")    
    show_word_synonyms(synonyms,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    print(100*"-")    
    print("if_occurrence\n")
    show_word_synonyms(if_occurence,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    print(100*"-")
    print("Cooccurrence\n")
    show_word_synonyms(cooccurence,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    print(100*"-")
    print("Client Comparison penalty = 0\n")
    show_word_synonyms(comparison_Matrix_0,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    print(100*"-")
    print("Client Comparison penalty = 0.1\n")
    show_word_synonyms(comparison_Matrix_01,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    print(100*"-")
    print("Client Comparison penalty = 0.0000000001\n")
    show_word_synonyms(comparison_Matrix_001,Produkt_lookup,product_index=product_index,n_synonyms=n_synonyms)
    return synonyms,if_occurence,cooccurence,comparison_Matrix_0,comparison_Matrix_01,comparison_Matrix_001,Produkt_lookup
    
def get_most_locrative_Products_for_price_reduction(show_n=10):
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_looku = get_Kunden_Produkte_Matrix()
    if_occurence,cooccurence,occurence=get_co_occurence_matrix(False)
    Avg_Menge,Avg_Rabatt,Avg_Gewinn,Avg_Gewinn_pro_stk=get_metadata()
    n_Produkte = len(occurence)
    most_locrative_produkts=sorted(zip(if_occurence.T.dot(Avg_Gewinn_pro_stk)*occurence,np.arange(n_Produkte)),reverse=True)
    for prod in most_locrative_produkts[:show_n]:
        print(100*"-")
        print(Produkt_lookup[prod[1]])
        for sugg in sorted(zip(if_occurence[:,prod[1]],np.arange(n_Produkte)),reverse = True)[:10]:
            print(sugg[0],Produkt_lookup[sugg[1]])
        
def PCA(x,reduction = 0):
    val,vec = np.linalg.eig(np.cov(x))
    pairs  = [(val[i],vec[i]) for i in range(np.shape(x)[0])]
    pairs.sort(reverse=True)
    M = pairs[0][1][:,np.newaxis]
    for n in range(np.shape(x)[0]-1-reduction):
        M = np.hstack((M,pairs[n+1][1][:,np.newaxis]))
    return x.T.dot(M).T

def compare_products(raw_string=r"Signal"): 
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_looku = get_Kunden_Produkte_Matrix()
    ret =[]
    for pattern in find_pattern(raw_string,Produkt_lookup):
        ret.append(Kunden_Produkte_Matrix[:,pattern[1]])
    filtered_ret=[]
    for col in np.array(ret).T:
        if np.sum(col) != 0:
            filtered_ret.append(col)
    return np.array(filtered_ret).T

def Cluster_and_evaluate(n_clusters=10):
    LSA_approx,Produkt_lookup = get_LSA_Transform()
    if_occurence,cooccurence,occurence=get_co_occurence_matrix(False)
    y=Cluster(cooccurence,n_clusters)
    #y=Cluster(LSA_approx.T,n_clusters)
    eval_ = [[] for i in range(n_clusters)]
    for i in range(n_clusters):
        for val in np.argwhere(y==i):
            eval_[i].append(Produkt_lookup[val[0]])
    return eval_
        
def make_training_data():
    ges_X=[]
    ges_T=[]
   
    Kunden_Produkte_Matrix,Produkt_lookup,Kunden_looku = get_Kunden_Produkte_Matrix()
    (n_Kunden,n_Produkte) = np.shape(Kunden_Produkte_Matrix)
    clients=sort_for_Clients() 
    Client_X_T= [[] for i in range(len(clients))]
    for client,client_index in zip(clients,np.arange(len(clients))): 
        X=[]
        T=[]
        for i in range(len(client)-1):
            x=np.zeros(n_Produkte)
            x[int(client[i][1])]=1
            X.append(x)
            ges_X.append(x)
            t=np.zeros(n_Produkte)
            t[int(client[i+1][1])]=1
            T.append(t)
            ges_T.append(t)
            
        Client_X_T[client_index]=(np.array(X),np.array(T))
            
    return Client_X_T,np.array(ges_X),np.array(ges_T)
            
        
    