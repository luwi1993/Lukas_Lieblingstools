# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:45:16 2019

@author: Lukas Widowski
"""
import numpy as np
def sort(array=[12,4,5,6,7,3,1,15]):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0][0]
        for x in array:
            if x[0] < pivot:
                less.append(x)
            elif x[0] == pivot:
                equal.append(x)
            elif x[0] > pivot:
                greater.append(x)
        
        return sort(less)+equal+sort(greater) 
    else: 
        return array
    
def reverse(array=[12,4,5,6,7,3,1,15]): 
    ret = []
    for i in range(len(array)):
        ret.append(array[-i-1])
    return ret 