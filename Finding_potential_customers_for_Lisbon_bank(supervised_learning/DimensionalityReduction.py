#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 19:12:50 2018

@author: ruchinpatel
"""

import numpy as np
import sklearn as sk
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt

def PCA(data,labels,np,what_dimensions):
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=what_dimensions)
    transformed_data = pca.fit(data).transform(data)
    print(pca.score(data))
    return transformed_data
        
        
        
    
    return transformed_data

def LDA(data,label,np,what_dimensions):
       
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=what_dimensions)
    transformed_data = lda.fit(data,label).transform(data)
    return transformed_data




