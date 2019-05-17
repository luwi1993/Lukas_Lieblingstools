# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:05:15 2019

@author: Lukas Widowski
"""

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);