#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:57:20 2018

@author: ruchinpatel
"""

import pandas as pd
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
import Cross_validation as p_CV
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin

class KernelDensityClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self,bandwidth = 3.3, kernel = 'gaussian',Lambda10=1,Lambda01=8,Lambda00=0,Lambda11=0):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.Lambda10 = Lambda10 
        self.Lambda01 = Lambda01
        self.Lambda00 = Lambda00
        self.Lambda11 = Lambda11
    
        
        
    def fit(self,data_train,labels_train):
        
        self.Lambda10 = np.log(self.Lambda10)
        self.Lambda01 = np.log(self.Lambda01)
        
        if((self.Lambda00 != 0) and (self.Lambda11 != 0)):
            self.Lambda00 = np.log(self.Lambda00)
            self.Lambda11 = np.log(self.Lambda11)
        elif((self.Lambda00) == 0 and (self.Lambda11 != 0)):
            self.Lambda00 = 0
            self.Lambda11 = np.log(self.Lambda11)
        elif((self.Lambda00 != 0) and (self.Lambda11 == 0)):
            self.Lambda00 = np.log(self.Lambda00)
            self.Lambda11 = 0
        else:
            self.Lambda00 = 0
            self.Lambda11 = 0
        
        self.unique_labels_train = np.sort(np.unique(labels_train))
        class_splitted_train_set = []
       
        X_0 = data_train[labels_train == self.unique_labels_train[0]] 
        X_1 = data_train[labels_train == self.unique_labels_train[1]]
        
        class_splitted_train_set.append(X_0)
        class_splitted_train_set.append(X_1)
        
        self.density_estimates = []
        estimator_0 = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel).fit(X_0)
        estimator_1 = KernelDensity(bandwidth=self.bandwidth,kernel=self.kernel).fit(X_1)
        self.density_estimates.append(estimator_0)
        self.density_estimates.append(estimator_1)
        
        self.class_wise_logpriors = []
        log_prior_0 = np.log(X_0.shape[0] / data_train.shape[0])
        log_prior_1 = np.log(X_1.shape[0] / data_train.shape[0])
        self.class_wise_logpriors.append(log_prior_0)
        self.class_wise_logpriors.append(log_prior_1)
        
        return self
    
    def pred_likelihood(self,data_test):
        
        
        logprobs_test_0 = self.density_estimates[0].score_samples(data_test)
        logprobs_test_1 = self.density_estimates[1].score_samples(data_test)
        
        result_0 = np.exp((self.Lambda10 - self.Lambda00) + self.class_wise_logpriors[0] + logprobs_test_0)
        result_1 = np.exp((self.Lambda01 - self.Lambda11) + self.class_wise_logpriors[1] + logprobs_test_1)
        
        result = np.c_[result_0,result_1]
        result = result / result.sum(1, keepdims=True)
        
        return result
    
    def predict(self, data_test):
        
        result = self.pred_likelihood(data_test)
        y_pred = np.empty((result.shape[0]))
    
        for i in range(0,result.shape[0]):
            b = result[i]
            yi = np.argmax(b)
            y_pred[i] = yi
    
        return y_pred

def plotRoc_Bayes(X,y,bandwidth,kernel,L10,L01,s1):
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42,stratify = y)
    
    kde = KernelDensityClassifier(bandwidth,kernel,L10,L01,0,0)
    kde.fit(X_train,y_train)
    likelihood = kde.pred_likelihood(X_test)
    confidence_score = np.empty((likelihood.shape[0]))
    
    for i in range(0,likelihood.shape[0]):
            b = likelihood[i]
            if(int(y[i]) == 0):
                idx = 1
            else:
                idx = 0
            confidence_score[i] = b[idx]
       
    y_pred = kde.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.figure()
    plt.title('F1 = %0.2f'%f1+' bw:'+str(bandwidth)+' L10:'+str(L10)+' L01'+str(L01))
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% area_curve)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    

def plotRoc_feat_wise_Bayes(X,y,bandwidth,kernel,L10,L01,plt):
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42,stratify = y)
    
    kde = KernelDensityClassifier(bandwidth,kernel,L10,L01,0,0)
    kde.fit(X_train,y_train)
    likelihood = kde.pred_likelihood(X_test)
    confidence_score = np.empty((likelihood.shape[0]))
    
    for i in range(0,likelihood.shape[0]):
            b = likelihood[i]
            if(int(y[i]) == 0):
                idx = 1
            else:
                idx = 0
            confidence_score[i] = b[idx]
       
    y_pred = kde.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.plot(false_positive_rate, true_positive_rate,label=['AUC = %0.2f'% area_curve,'F1 = %0.2f'%f1,'feats = %d'% X_train.shape[1]])      
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    
    return area_curve
    
        

def plotRoc_feat_wise_SVM(X,y,c,plt,g='auto'):
    
    X_dim_red  = X
    
    X_train, X_test, y_train, y_test =  train_test_split(X_dim_red, y, test_size=0.1, random_state=42)
    
    
    clf = SVC(C=c,kernel = 'rbf',class_weight = {1:5},gamma = g)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    confidence_score = clf.decision_function(X_test)
    confidence_score = preprocessing.scale(confidence_score)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
#    plt.figure()
#    plt.title('ROC and F1 = %0.2f'%f1)
    plt.plot(false_positive_rate, true_positive_rate,label=['AUC = %0.2f'% area_curve,'F1 = %0.2f'%f1,'feats = %d'% X_train.shape[1]])      
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    
    return area_curve
    

def plotRoc_feat_wise_percep(X,y,class_w,plt,m_itr=None):
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42)
    perceptron = Perceptron(class_weight = {1: class_w},max_iter = m_itr).fit(X_train,y_train)
    predicted_labels_test = perceptron.predict(X_test)
    
    confidence_score = perceptron.decision_function(X_test)
    confidence_score = preprocessing.scale(confidence_score)
    
    f1 = f1_score(y_test, predicted_labels_test)
    precision = precision_score(y_test,predicted_labels_test)
    recall = recall_score(y_test,predicted_labels_test)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.plot(false_positive_rate, true_positive_rate,label=['AUC = %0.2f'% area_curve,'F1 = %0.2f'%f1,'feats = %d'% X_train.shape[1]])
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    
    return area_curve
    




def plotRocCurve_Unbalanced(X,y,class_w,s1,m_itr=None):
    
    print()
    print()
    print("IN plotROC UNBALANCED")
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42)
    perceptron = Perceptron(class_weight = {1: class_w},max_iter = m_itr).fit(X_train,y_train)
    predicted_labels_train = perceptron.predict(X_train)
    predicted_labels_test = perceptron.predict(X_test)
    
    confidence_score = perceptron.decision_function(X_test)
    confidence_score = preprocessing.scale(confidence_score)
    
    f1 = f1_score(y_test, predicted_labels_test)
    precision = precision_score(y_test,predicted_labels_test)
    recall = recall_score(y_test,predicted_labels_test)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.figure()
    plt.title('ROC and F1 = %0.2f'%f1+' class_weight:'+str(class_w)+' max_iter:'+str(m_itr))
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% area_curve)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
def plotRocCurve_balanced(X,y,s1,m_itr):
    
    print()
    print()
    print("IN plotROC BALANCED")
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.1, random_state=42)
    
    balancing_obj = SMOTE(random_state=42,k_neighbors=5)
    X_train, y_train = balancing_obj.fit_sample(X_train, y_train)
    
    perceptron = Perceptron(max_iter = m_itr).fit(X_train,y_train)
    predicted_labels_train = perceptron.predict(X_train)
    predicted_labels_test = perceptron.predict(X_test)
    
    confidence_score = perceptron.decision_function(X_test)
    confidence_score = preprocessing.scale(confidence_score)
    
    f1 = f1_score(y_test, predicted_labels_test)
    precision = precision_score(y_test,predicted_labels_test)
    recall = recall_score(y_test,predicted_labels_test)
    print('precision is: ',precision)
    print('recall is:',recall)
    print('f1 score is :',f1)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.figure()
    plt.title('ROC and F1 = %0.2f'%f1+' class_weight:'+' max_iter:'+str(m_itr))
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% area_curve)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    #plt.xlim([0,1])
    #plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
    

    
    