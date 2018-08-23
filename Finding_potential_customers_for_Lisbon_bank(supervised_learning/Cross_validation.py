#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:46:06 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:32:44 2018

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
import DimensionalityReduction as dim_red
from sklearn.feature_selection import RFE
import plotRocCurve as plt_roc
from sklearn.neighbors import KernelDensity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit


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


def CVFeature_sel_Bayes(data,labels,s1,s2):
    feature_list = list(np.arange(1,data.shape[1]+1,1))
    F1 = []
    AUC = []
    f1_scorer = make_scorer(f1_score,pos_label=1)
    bandwidths = np.arange(1,4,0.1)
    f_scores_per_feature = dict()
    
    for f_size in feature_list:
        print(f_size)
        estimator = SVC(kernel="linear")
#        estimator = KernelDensityClassifier()
        selector = RFE(estimator, f_size, step=1)
        selector = selector.fit(data,labels)
        data_new = data[:,list(selector.support_)]
        X_train, X_test, y_train, y_test =  train_test_split(data_new, labels, test_size=0.1, random_state=42)
        grid = GridSearchCV(KernelDensityClassifier(),{'bandwidth': bandwidths},scoring=f1_scorer,n_jobs=-1)
        grid.fit(X_train,y_train)
        f1 = grid.best_score_
        F1.append(f1)
    
        f_scores_per_feature[(f_size,f1)] = selector.support_
        AUC.append(plt_roc.plotRoc_feat_wise_Bayes(data_new,labels,grid.best_params_.get('bandwidth'),'gaussian',1,8,plt))
        
    
    plt.title('ROC curves for 25 features selected from 1 to 25 using SVM RBF kernel')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
    width = 0.25       # the width of the bars
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    rects1 = ax.bar(feature_list, F1, width, color='goldenrod')
    
    rects2 = ax.bar(np.array(feature_list)+width,AUC, width, color='teal')
    
    # add some
    t = np.core.defchararray.add(list(map(str, feature_list)),'')
    ax.set_ylabel('F1 scores and AUC scales')
    ax.set_title('F1 scores and ROC by total features selected')
    ax.set_xticks(np.array(feature_list) + width / 2)
    ax.set_xticklabels(tuple(t),fontdict = {'fontsize': 6})
#    ax.xticks(feature_list, (t))
    ax.set_yticks(np.arange(0, 1, 0.2))
    
    ax.legend( (rects1[0], rects2[0]), ('F1', 'AUC') )
    
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(s2,dpi=300)
    return f_scores_per_feature
        
    

def CVFeatureSel_SVM(data,labels,s1,s2):
    feature_list = list(np.arange(1,data.shape[1]+1,1))
    
    f_scores_per_feature = dict()
    F1 = []
    AUC = []
    
    for f_size in feature_list:
        
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, f_size, step=1)
        selector = selector.fit(data,labels)
        data_new = data[:,list(selector.support_)]
#        dim_estimators_NA = CVPerceptronfinal(data_new,labels,f_size)
        X_train, X_test, y_train, y_test =  train_test_split(data_new, labels, test_size=0.1, random_state=42)
#        perceptron = Perceptron(class_weight = {1: 8},max_iter = 50).fit(X_train,y_train)
        clf = SVC(C=800,kernel ='rbf',class_weight = {1:5}).fit(X_train,y_train)
        predicted_labels_test = clf.predict(X_test)
        f1 = f1_score(y_test, predicted_labels_test)
        F1.append(f1)
#        precision = precision_score(y_test,predicted_labels_test)
#        recall = recall_score(y_test,predicted_labels_test)
        f_scores_per_feature[(f_size,f1)] = selector.support_
        AUC.append(plt_roc.plotRoc_feat_wise_SVM(data_new,labels,800,plt))
        
    plt.title('ROC curves for 25 features selected from 1 to 25 using SVM RBF kernel')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
    width = 0.25       # the width of the bars
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    rects1 = ax.bar(feature_list, F1, width, color='royalblue')
    
    rects2 = ax.bar(np.array(feature_list)+width,AUC, width, color='seagreen')
    
    # add some
    t = np.core.defchararray.add(list(map(str, feature_list)),'')
    ax.set_ylabel('F1 scores and AUC scales')
    ax.set_title('F1 scores and ROC by total features selected')
    ax.set_xticks(np.array(feature_list) + width / 2)
    ax.set_xticklabels(tuple(t),fontdict = {'fontsize': 6})
#    ax.xticks(feature_list, (t))
    ax.set_yticks(np.arange(0, 1, 0.2))
    
    ax.legend( (rects1[0], rects2[0]), ('F1', 'AUC') )
    
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(s2,dpi=300)
    return f_scores_per_feature

def CVFeatureSel_Percep(data,labels,s1,s2):
    feature_list = list(np.arange(1,data.shape[1]+1,1))
    
    f_scores_per_feature = dict()
    
    f_scores_per_feature = dict()
    F1 = []
    AUC = []
    
    for f_size in feature_list:
        
        estimator = SVC(kernel="linear")
        selector = RFE(estimator, f_size, step=1)
        selector = selector.fit(data,labels)
        data_new = data[:,list(selector.support_)]
#        dim_estimators_NA = CVPerceptronfinal(data_new,labels,f_size)
        X_train, X_test, y_train, y_test =  train_test_split(data_new, labels, test_size=0.1, random_state=42)
        perceptron = Perceptron(class_weight = {1: 8},max_iter = 50).fit(X_train,y_train)
        predicted_labels_test = perceptron.predict(X_test)
        f1 = f1_score(y_test, predicted_labels_test)
        F1.append(f1)
#        precision = precision_score(y_test,predicted_labels_test)
#        recall = recall_score(y_test,predicted_labels_test)
        f_scores_per_feature[(f_size,f1)] = selector.support_
        AUC.append(plt_roc.plotRoc_feat_wise_percep(data_new,labels,8,plt,50))
    
   
    plt.title('ROC curves for 25 features selected from 1 to 25 using Perceptron')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
    width = 0.25       # the width of the bars
    
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    rects1 = ax.bar(feature_list, F1, width, color='#624ea7')
    
    rects2 = ax.bar(np.array(feature_list)+width,AUC, width, color='red')
    
    # add some
    t = np.core.defchararray.add(list(map(str, feature_list)),'')
    ax.set_ylabel('F1 scores and AUC scales')
    ax.set_title('F1 scores and ROC by total features selected')
    ax.set_xticks(np.array(feature_list) + width / 2)
    ax.set_xticklabels(tuple(t),fontdict = {'fontsize': 6})
#    ax.xticks(feature_list, (t))
    ax.set_yticks(np.arange(0, 1, 0.2))
    
    ax.legend( (rects1[0], rects2[0]), ('F1', 'AUC') )
    
    fig2 = plt.gcf()
    plt.show()
    fig2.savefig(s2,dpi=300)
    
    return f_scores_per_feature
    



def CVPerceptronDim(data,labels):
    
    print()
    print()
    print("INSIDE CVPERCEPTRONDIM")
    PCA_dimensions = list(np.arange(data.shape[1],0,-1))
    percep_class_weight = []
    for i in range(1,11):
        percep_class_weight.append({1:i})    
    max_iterations = list(range(1,100,2))
    parameters = {'max_iter':max_iterations, 'class_weight':percep_class_weight}
    f1_scorer = make_scorer(f1_score)
    F1_dim = dict()
    
    for dim in PCA_dimensions:
        
        X_dim_red  = dim_red.PCA(data,labels,np,dim) 
        skf1 = StratifiedKFold(n_splits = 5, shuffle = True)
        
        F1 = []
        estimator_weight = []
        estimator_max_itr = []
        for train_index, test_index in skf1.split(X_dim_red,labels):
            X_err_train, X_err_test = X_dim_red[train_index], X_dim_red[test_index]
            y_err_train, y_err_test = labels[train_index], labels[test_index]
            
#            percep_CV = Perceptron(random_state=42)
#            clf = GridSearchCV(percep_CV, param_grid=parameters,scoring=f1_scorer,n_jobs=-1,cv=5)
#            clf.fit(X_err_train, y_err_train)
            
            clf = CVPerceptronfinal(X_err_train,y_err_train,dim)
            print()
            print()
            print("INSIDE CVPERCEPTRONDIM after clf")
            
            print('best f1 score is for Grid CV is',clf.get('avg_f1'))
            print('best class_weight is:',clf.get('best_weight'))
            print('best max_iter is:',clf.get('best_max_iter'))
            
            percep_test = Perceptron(max_iter = clf.get('best_max_iter'),class_weight = {1:clf.get('best_weight')})
            percep_test.fit(X_err_train, y_err_train)
            y_pred = percep_test.predict(X_err_test)
            f1_test = f1_score(y_err_test,y_pred)
            print("Test f1 score is:",f1_test)
            F1.append(f1_test)
            estimator_weight.append(clf.get('best_weight'))
            estimator_max_itr.append(clf.get('best_max_iter'))
            
        
        idx = np.argmax(F1)
        final_est = {'avg_f1':np.mean(F1),'best_weight':estimator_weight[idx],'best_max_iter':estimator_max_itr[idx]}
        F1_dim[dim] = final_est
        
    return F1_dim

            
def CVNeuralnet(data,labels,s1):
    
    X_train, X_test, y_train, y_test = train_test_split(data,labels)
    
    balancing_obj = SMOTE(random_state=42,k_neighbors=5)
    X_train, y_train = balancing_obj.fit_sample(data, labels)
    
    max_itr = list(np.arange(1,60,3))
    hidden_layers = [(25,5),(25,10,5),(25,100,20,10,5),(25,100,40,20,10,5),(25,100,50,40,20,10,5)]
    a = [0.0001,0.0002,0.0003,0.0004,0.0005]
    

    
    
    parameters = {'hidden_layer_sizes':hidden_layers,'max_iter':max_itr,'alpha':a}
    
    f1_scorer = make_scorer(f1_score,pos_label=1)
    
    grid = GridSearchCV(MLPClassifier(activation ='relu',random_state = 10),param_grid = parameters,scoring=f1_scorer,n_jobs=-1)
    grid.fit(X_train,y_train)
    
    f1 = grid.best_score_
    best_hidden_layers = grid.best_params_.get('hidden_layer_sizes')
    best_max_iter = grid.best_params_.get('max_iter')
    best_alpha = grid.best_params_.get('alpha')
    
    str_hidden = '('
    for i in range(len(best_hidden_layers)):
        str_hidden = str_hidden + str(best_hidden_layers[i]) + ','
    str_hidden = str_hidden + ')'
        
    
    mlp = MLPClassifier(hidden_layer_sizes=best_hidden_layers,max_iter=best_max_iter,activation='relu',random_state = 10,alpha=best_alpha)

    mlp.fit(X_train,y_train)
    
    y_pred = mlp.predict(X_test)
    
    likelihood = mlp.predict_proba(X_test)
    confidence_score = np.empty((likelihood.shape[0]))
    
    f1 = f1_score(y_test,y_pred)
        
    for i in range(0,likelihood.shape[0]):
        b = likelihood[i]
        if(int(y_test[i]) == 0):
            idx = 1
        else:
            idx = 0
        confidence_score[i] = b[idx]
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, confidence_score,pos_label = 1)
    area_curve = auc(false_positive_rate, true_positive_rate)
    
    plt.figure()
    plt.title('F1 = %0.2f'%f1+' max_iter:'+str(best_max_iter)+' alpha:'+str(best_alpha)+' hdn_lyrs: '+str_hidden)
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
    
    
    


def CVPerceptronfinal(data,labels,dim=0):
    
    percep_class_weight = [2,3,4,5,6,7,8,9,10]
    max_iterations = list(range(1,100,2))
    if(dim != 0 and dim >= data.shape[1]):
        X_dim_red  = dim_red.PCA(data,labels,np,dim)
    else:
        X_dim_red  = data
    skf1 = StratifiedKFold(n_splits = 5, shuffle = True)
#    X_dim_red = data
#    F1 = []
    F1_T = []
    weight_T = []
    max_iter_T = []
    mean_cv_f1 = []
    mean_cv_std = []
    for train_err_index, test_err_index in skf1.split(X_dim_red,labels):
        
        X_err_train, X_err_test = X_dim_red[train_err_index], X_dim_red[test_err_index]
        y_err_train, y_err_test = labels[train_err_index], labels[test_err_index]
        F1 = np.zeros((len(percep_class_weight),len(max_iterations)))
        DEV = np.zeros((len(percep_class_weight),len(max_iterations)))
        
        final_weight = 0
        final_max_iter = 0
        for i in range(len(percep_class_weight)):
            for j in range(len(max_iterations)):
                weight = percep_class_weight[i]
                m_itr = max_iterations[j]
                
                skf = StratifiedKFold(n_splits = 5, shuffle = True)
                all_f1 = []
                for train_index, dev_index in skf.split(X_err_train,y_err_train):
                    X_cv_train, X_cv_dev = X_err_train[train_index], X_err_train[dev_index]
                    y_cv_train, y_cv_dev = y_err_train[train_index], y_err_train[dev_index]
                    clf = Perceptron(max_iter = m_itr,class_weight={1:weight},random_state=42)
                    clf.fit(X_cv_train,y_cv_train)
                    y_pred = clf.predict(X_cv_dev)
                    f1 = f1_score(y_cv_dev,y_pred)
                    all_f1.append(f1)
                    
                mean = np.mean(all_f1)
                s_dev = np.std(all_f1)
                
                F1[i,j] = mean
                DEV[i,j] = s_dev
                
        min_mean_indices = np.argwhere(F1 == np.max(F1))
        
        
        if(min_mean_indices.shape[0] == 1):
            print("Inside mean")
            index = min_mean_indices[0]
    #        print(index)
            print("mean cross validation f1 is: ",F1[index[0],index[1]])
            print("Standard Deviation is: ",DEV[index[0],index[1]])
            mean_cv_f1.append(F1[index[0],index[1]])
            mean_cv_std.append(DEV[index[0],index[1]])
            print("Value of weight corrosponding to this is: ",percep_class_weight[index[0]])
            print("Value of maximum iterations corrosponding to this is: ",max_iterations[index[1]])
            final_weight = percep_class_weight[index[0]]
            final_max_iter = max_iterations[index[1]]
            print()
        else:
            st_dev_key = dict()
            for l in range(0,min_mean_indices.shape[0]):
                row = min_mean_indices[l,0]
                col = min_mean_indices[l,1]
                print("F1 scores are: ",F1[row,col])
                st_dev_key[(row,col)] = DEV[row,col]
            
        
            print("Inside Stdev")
    #        min_dev_indices = np.argwhere(DEV == np.min(DEV))
    #        sel_index = np.random.choice(np.array(range(0,min_dev_indices.shape[0])))
    #        index = min_dev_indices[sel_index]
            index = min(st_dev_key, key=st_dev_key.get)
    #        print(index)
            print("mean cross validation F1 score is: ",F1[index[0],index[1]])
            print("Standard Deviation is: ",DEV[index[0],index[1]])
            mean_cv_f1.append(F1[index[0],index[1]])
            mean_cv_std.append(DEV[index[0],index[1]])
            print("Value of weight corrosponding to this is: ",percep_class_weight[index[0]])
            print("Value of maximum iterations corrosponding to this is: ",max_iterations[index[1]])
            final_weight = percep_class_weight[index[0]]
            final_max_iter = max_iterations[index[1]]
            print()
        
        
        
        clf1 = Perceptron(class_weight={1:final_weight},max_iter = final_max_iter,random_state=42)
        clf1.fit(X_err_train,y_err_train.ravel(y_err_train.shape[0],))
        y_pred_test = clf.predict(X_err_test)
        f1_test = f1_score(y_err_test,y_pred_test)
        
        
        if(f1_test != 0):
            F1_T.append(f1_test)
            weight_T.append(final_weight)
            max_iter_T.append(final_max_iter)
        
    
    print(F1_T)
    print(weight_T)
    print(max_iter_T)
    
    print("Mean f1 of 5 fold cross validation is: ",np.mean(mean_cv_f1))
    print("Standard deviation is:",np.mean(mean_cv_std))
    
    idx = np.argmax(F1_T)
    final_est = {'avg_f1':np.mean(F1_T),'best_weight':weight_T[idx],'best_max_iter':max_iter_T[idx]}
    
    return final_est




def CVPerceptronfinal_balanced(data,labels,dim=0):
    
    max_iterations = list(range(1,100,2))
    if(dim != 0 and dim >= data.shape[1]):
        X_dim_red  = dim_red.PCA(data,labels,np,dim)
    else:
        X_dim_red  = data
    skf1 = StratifiedKFold(n_splits = 5, shuffle = True)
#    X_dim_red = data
#    F1 = []
    F1_T = []
    max_iter_T = []
    mean_cv_f1 = []
    mean_cv_std = []
    for train_err_index, test_err_index in skf1.split(X_dim_red,labels):
        
        X_err_train, X_err_test = X_dim_red[train_err_index], X_dim_red[test_err_index]
        y_err_train, y_err_test = labels[train_err_index], labels[test_err_index]
        F1 = np.zeros((len(max_iterations)))
        DEV = np.zeros((len(max_iterations)))
        
        balancing_obj = SMOTE(random_state=42,k_neighbors=5)
        X_err_train_balanced, y_err_train_balanced = balancing_obj.fit_sample(X_err_train, y_err_train)
        
        final_max_iter = 0
        
        for j in range(len(max_iterations)):
            m_itr = max_iterations[j]
            
            skf = StratifiedKFold(n_splits = 5, shuffle = True)
            all_f1 = []
            for train_index, dev_index in skf.split(X_err_train,y_err_train):
                X_cv_train, X_cv_dev = X_err_train[train_index], X_err_train[dev_index]
                y_cv_train, y_cv_dev = y_err_train[train_index], y_err_train[dev_index]
                
                balancing_obj = SMOTE(random_state=42,k_neighbors=5)
                X_cv_train, y_cv_train = balancing_obj.fit_sample(X_cv_train, y_cv_train)
                
                clf = Perceptron(max_iter = m_itr,random_state=42)
                clf.fit(X_cv_train,y_cv_train.ravel(y_cv_train.shape[0],))
                y_pred = clf.predict(X_cv_dev)
                f1 = f1_score(y_cv_dev,y_pred)
                all_f1.append(f1)
                
            mean = np.mean(all_f1)
            s_dev = np.std(all_f1)
            
            F1[j] = mean
            DEV[j] = s_dev
                
        min_mean_indices = np.argwhere(F1 == np.max(F1))
        
        
        if(min_mean_indices.shape[0] == 1):
            print("Inside mean")
            index = min_mean_indices[0]
    #        print(index)
            print("mean cross validation f1 is: ",F1[index[0]])
            print("Standard Deviation is: ",DEV[index[0]])
            mean_cv_f1.append(F1[index[0]])
            mean_cv_std.append(DEV[index[0]])
            print("Value of maximum iterations corrosponding to this is: ",max_iterations[index[0]])
            final_max_iter = max_iterations[index[0]]
            print()
        else:
            st_dev_key = dict()
            for l in range(0,min_mean_indices.shape[0]):
                col = min_mean_indices[l]
                print("F1 scores are: ",F1[col])
                st_dev_key[col] = DEV[col]
            
        
            print("Inside Stdev")
    #        min_dev_indices = np.argwhere(DEV == np.min(DEV))
    #        sel_index = np.random.choice(np.array(range(0,min_dev_indices.shape[0])))
    #        index = min_dev_indices[sel_index]
            index = min(st_dev_key, key=st_dev_key.get)
    #        print(index)
            print("mean cross validation F1 score is: ",F1[index[0]])
            print("Standard Deviation is: ",DEV[index[0]])
            mean_cv_f1.append(F1[index[0]])
            mean_cv_std.append(DEV[index[0]])
            print("Value of maximum iterations corrosponding to this is: ",max_iterations[index[0]])
            final_max_iter = max_iterations[index[1]]
            print()
        
        
        print('here 1')
        clf1 = Perceptron(max_iter = final_max_iter,random_state=42)
        clf1.fit(X_err_train_balanced,y_err_train_balanced.ravel(y_err_train_balanced.shape[0],))
        y_pred_test = clf.predict(X_err_test)
        f1_test = f1_score(y_err_test,y_pred_test)
        
        
        if(f1_test != 0):
            F1_T.append(f1_test)
            max_iter_T.append(final_max_iter)
        
    
    print(F1_T)
    print(max_iter_T)
    
    print("Mean f1 of 5 fold cross validation is: ",np.mean(mean_cv_f1))
    print("Standard deviation is:",np.mean(mean_cv_std))
    
    idx = np.argmax(F1_T)
    final_est = {'avg_f1':np.mean(F1_T),'best_max_iter':max_iter_T[idx]}
    
    return final_est   
       
        
def CVSVM_Unbalanced(data,labels,s1,dim=None):
    
    if(dim !=  None):
        X_dim_red  = dim_red.PCA(data,labels,np,dim)
    else:
        X_dim_red = data
    
    C_s = np.logspace(-3,3,10)
    gamma_s  = np.logspace(-3,3,10)
    parameters = dict(gamma=gamma_s, C=C_s)
    f1_scorer = make_scorer(f1_score,pos_label=1)
    X_err_train, X_err_test, y_err_train, y_err_test =  train_test_split(X_dim_red, labels, test_size=0.1, random_state=42,stratify = labels)
    
    skf1 = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    grid_cv = GridSearchCV(SVC(class_weight = {1:8}), param_grid=parameters,scoring=f1_scorer,n_jobs=-1,cv=skf1)
    grid_cv.fit(X_err_train,y_err_train)
    print("Mean cross Validated f1 score is:",grid_cv.best_score_)
    print("Best C is:",grid_cv.best_params_.get('C'))
    print("Best gamma is:",grid_cv.best_params_.get('gamma'))
    
    C_best = grid_cv.best_params_.get('C')
    gamma_best = grid_cv.best_params_.get('gamma')
    clf = SVC(kernel='rbf',class_weight ={1:5},C = C_best,gamma = gamma_best,probability=True,random_state=42)
    
    probas_ = clf.fit(X_err_train, y_err_train).predict_proba(X_err_test)
    y_pred = clf.fit(X_err_train, y_err_train).predict(X_err_test)
    f1 = f1_score(y_err_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_err_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%roc_auc)
    plt.plot([0, 1], [0, 1], '--', color='r', label='random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC, test F1:%0.2f'%f1+' C:'+str(grid_cv.best_params_.get('C'))+' gamma:'+str(grid_cv.best_params_.get('gamma'))+' mean_f1_cv:'+str(grid_cv.best_score_))
    plt.legend(loc="lower right")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
    
    
def CVSVM_balanced(data,labels,s1,dim=None):
    
    if(dim !=  None):
        X_dim_red  = dim_red.PCA(data,labels,np,dim)
    else:
        X_dim_red = data
    
    C_s = np.logspace(-3,3,10)
    gamma_s  = np.logspace(-3,3,10)
    parameters = dict(gamma=gamma_s, C=C_s)
    f1_scorer = make_scorer(f1_score,pos_label=1)
    
    
    balancing_obj = SMOTE(random_state=42,k_neighbors=5)
    X_dim_red, labels= balancing_obj.fit_sample(X_dim_red,labels)
    
    X_err_train, X_err_test, y_err_train, y_err_test =  train_test_split(X_dim_red, labels, test_size=0.1, random_state=42)
    
    
    skf1 = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    grid_cv = GridSearchCV(SVC(class_weight = {1:8}), param_grid=parameters,scoring=f1_scorer,n_jobs=-1,cv=skf1)
    grid_cv.fit(X_err_train,y_err_train)
    print("Mean cross Validated f1 score is:",grid_cv.best_score_)
    print("Best C is:",grid_cv.best_params_.get('C'))
    print("Best gamma is:",grid_cv.best_params_.get('gamma'))
    
    C_best = grid_cv.best_params_.get('C')
    gamma_best = grid_cv.best_params_.get('gamma')
    clf = SVC(kernel='rbf',class_weight ={1:5},C = C_best,gamma = gamma_best,probability=True,random_state=42)
    
    probas_ = clf.fit(X_err_train, y_err_train).predict_proba(X_err_test)
    y_pred = clf.fit(X_err_train, y_err_train).predict(X_err_test)
    f1 = f1_score(y_err_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_err_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%roc_auc)
    plt.plot([0, 1], [0, 1], '--', color='r', label='random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC, test F1:%0.2f'%f1+' C:'+str(grid_cv.best_params_.get('C'))+' gamma:'+str(grid_cv.best_params_.get('gamma'))+' mean_f1_cv:'+str(grid_cv.best_score_))
    plt.legend(loc="lower right")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(s1,dpi=300)
   
    

              

            
         

            
            
            
                    
                    

