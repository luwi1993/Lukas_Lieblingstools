
"""
Created on Tue Apr  2 16:38:51 2019

@author: Lukas Widowski
"""

import pandas as pd
import numpy as np
import Test_preprocess as pre
import LSA 
import BAGofWords as bow
from operator import itemgetter, attrgetter, methodcaller
import matplotlib.pyplot as plt 
import re
import datetime
from IPython import get_ipython
plt.interactive(False)
get_ipython().run_line_magic('matplotlib', 'qt')

table_name="deutsche_bahn"
corpus=pd.read_csv("..\Processed_German_Corpus.csv").values
tweets=pd.read_csv("csv_exports/sst/"+table_name+".csv").values[:,1]
word_list=pd.read_csv("csv_exports/wil/"+table_name+".csv").values
sorted_word_list=sorted(word_list, key=itemgetter(2),reverse=True)



#min_appearance = 70
#whole_corpus=list(corpus[:,0])
#unscored_words=[]
#for line in sorted_word_list[:]:
#    if line[2]<min_appearance:
#        break
#    whole_corpus.append(line[0])
#    unscored_words.append(line[0])
#    
#reduced_corpus = []    
#for tweet in tweets:
#    for token in bow.preprocess(tweet):
#        if token in whole_corpus and token not in reduced_corpus:
#            reduced_corpus.append(token)


unscored_words=[]
min_appearance = 15
reduced_corpus = []
for line in sorted_word_list[:]:
    if line[2]<min_appearance:
        break
    reduced_corpus.append(line[0])
    unscored_words.append(line[0])
   


vektorized_tweets=np.zeros((len(tweets),len(reduced_corpus)))
for tweet_index in range(len(tweets)):
    for token in bow.preprocess(tweets[tweet_index]): 
        for word_index in range(len(reduced_corpus)):
            if token == reduced_corpus[word_index]:
                vektorized_tweets[tweet_index][word_index]+=1


vektorized_tweets = vektorized_tweets.T

n_words=len(reduced_corpus)
n_tweets=len(tweets)

def get_synonyms(vektorized_tweets,rank=0,do_find_best_rank=True,plot_it=True):
    LSA_approx,U,S,V= LSA.LSA(vektorized_tweets,rank,do_find_best_rank=do_find_best_rank,plot_it=plot_it)
    synonyms=np.array(U.dot(U.T))
    return synonyms


def show_n_synonyms(synonyms,reduced_corpus,n_synonyms=3,show_n=0):
    if show_n == 0:
        show_n = len(synonyms)
    for row in range(min(show_n,len(synonyms))):
        sorted_row=sorted(zip(synonyms[row],np.arange(len(synonyms))),reverse=True)
        print()
        print(row,reduced_corpus[row],bow.score_word_with_corpus(reduced_corpus[row],corpus))
        score = 0
        for top in range(n_synonyms):
            score += bow.score_word_with_corpus(reduced_corpus[sorted_row[top][1]],corpus)
            print(sorted_row[top][0],reduced_corpus[sorted_row[top][1]],bow.score_word_with_corpus(reduced_corpus[sorted_row[top][1]],corpus))
        print(20*"--")
        for top in range(n_synonyms):
            top+=1
            score-=bow.score_word_with_corpus(reduced_corpus[sorted_row[-top][1]],corpus)
            print(sorted_row[-top][0],reduced_corpus[sorted_row[-top][1]],bow.score_word_with_corpus(reduced_corpus[sorted_row[-top][1]],corpus))
        print("score:",score)
        
        
def show_synonyms_of_unscored_words(vektorized_tweets,unscored_words,reduced_corpus):
    synonyms=get_synonyms(vektorized_tweets)
    for word in unscored_words:
        show_synonyms(word,synonyms,reduced_corpus,5)
        

def show_synonyms(word,synonyms,reduced_corpus,n_synonyms=3):

    for index in range(len(reduced_corpus)):
        if word == reduced_corpus[index]:
            row = index
          
    sorted_row=sorted(zip(synonyms[row],np.arange(len(synonyms))),reverse=True)
    print()
    print(row,reduced_corpus[row],bow.score_word_with_corpus(reduced_corpus[row],corpus))
    for top in range(n_synonyms):
        print(sorted_row[top][0],reduced_corpus[sorted_row[top][1]],bow.score_word_with_corpus(reduced_corpus[sorted_row[top][1]],corpus))
#    for top in range(n_synonyms):
#        top+=1
#        print(sorted_row[-top][0],reduced_corpus[sorted_row[-top][1]],bow.score_word_with_corpus(reduced_corpus[sorted_row[-top][1]],corpus))


def show_synonym_sum(synonyms,reduced_corpus,print_out=False):
    ret = []
    for row in range(len(synonyms)):
        if print_out:
            print()
            print(row,reduced_corpus[row])
            print("score:",sum(synonyms[row]))
        ret.append([row,reduced_corpus[row],sum(synonyms[row]),bow.score_word_with_corpus(reduced_corpus[row],corpus)])
    ret=np.array(ret)    
    sorted_ret = sorted(zip(ret[:,2],ret[:,1],ret[:,0],ret[:,3]))
    return sorted_ret
    

def show(l=[]):
    for i in l:
        print(reduced_corpus[i],bow.score_word_with_corpus(reduced_corpus[i],corpus))

def show_change_LSA_ranks(vektorized_tweets,reduced_corpus,Interval=[0,0],step_size=5):
    if Interval == [0,0]:
        Interval[0]=0
        Interval[1]=len(reduced_corpus)
    
    if Interval[1] > len(reduced_corpus):
        Interval[1]=len(reduced_corpus)
        
    max_=-1000
    LSA_diff = []
    for i in np.linspace(Interval[0],Interval[1],(Interval[1]-Interval[0])/step_size):
        i = int(i)
        LSA_approx=LSA.LSA(vektorized_tweets,i)[0]
        diff = sum(sum(LSA_approx-vektorized_tweets))

        print(i,":",diff)
        LSA_diff.append([i,diff]) 
        if max_<diff:
            max_=diff
            max_i = i

    plt.plot(np.array(LSA_diff)[:,0],np.array(LSA_diff  )[:,1])
    return LSA_diff,max_i

