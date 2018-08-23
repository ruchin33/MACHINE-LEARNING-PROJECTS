#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:45:04 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 21:16:59 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:44:32 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 20:20:08 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:37:01 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:16:27 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 00:18:33 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:17:09 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:17:30 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:06:41 2018

@author: ruchinpatel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 17:27:37 2018

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
import plotRocCurve as plt_roc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import DimensionalityReduction as dim_red
#import SVMCV as svm_cv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity




#job is an ordered categorical data as an unemployed is not likely to open a bank account as is the entrepreneur
#Or in other words the jobs specifies the social standing of a person just like social status i.e poor average rich
#1: unemployed
#2: student
#3: housemaid
#4: retired
#5: self-employed
#6: admin. 
#7: services
#8: technician
#9: blue-collar
#10: management
#11: entrepreneur

#Level of education is also an ordered categorical variable as it gives the intellectual standing of a person
#1: illetrate
#2: basic.4y
#3: basic.6y
#4: basic.9y
#5: high.school
#6: professional.course
#7: university.degree

#Month is an ordered categorical variable 
#1: mar
#2: apr
#3: may
#4: jun
#5: jul
#6: aug
#7: sep
#8: oct
#9: nov
#10: dec

#Day_of_week is also an ordered categorical variable
#1: mon
#2: tue
#3: wed
#4: thu
#5: fri


#Marital is unordered
#1: divorced
#2: single
#3: married

#Housing
#1: no
#2: yes

#Loan
#1: no
#2: yes

#Contact
#1: telephone
#2: cellular

#Loan
#1: nonexistent
#2: failure
#3: success



    
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
    
    


class PreProcessData:
    
    
    def removeFeature(self,data,feature_name,pd,np):
        
        data = data.drop([feature_name],axis = 1)
        return data
    
    
    def AssignValsOrderedCategorical(self,data,pd,np,ordered_vals_dict,ordered_numeric_dict):
        
        
        for key in ordered_vals_dict:
            
            data[key] = data[key].replace(ordered_vals_dict.get(key),ordered_numeric_dict.get(key))
        
#        data['job'] = data['job'].replace(['unemployed','student','housemaid','retired','self-employed','admin.','services','technician','blue-collar','management','entrepreneur'],[1,2,3,4,5,6,7,8,9,10,11])
#        data['education'] = data['education'].replace(['illiterate','basic.4y','basic.6y','basic.9y','high.school','professional.course','university.degree'],[1,2,3,4,5,6,7])
#        data['month'] = data['month'].replace(['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],[1,2,3,4,5,6,7,8,9,10])
#        data['day_of_week'] = data['day_of_week'].replace(['mon','tue','wed','thu','fri'],[1,2,3,4,5])
            
        return data
    
    
    def AssignValsUnorderedCategorical(self,data,pd,np,unordered_vals_dict,unordered_numeric_dict):
        
        
        for key in unordered_vals_dict:
            
            data[key] = data[key].replace(unordered_vals_dict.get(key),unordered_numeric_dict.get(key))
        
#        data['marital'] = data['marital'].replace(['divorced','single','married'],[1,2,3])
#        data['housing'] = data['housing'].replace(['no','yes'],[1,2])
#        data['loan'] = data['loan'].replace(['no','yes'],[1,2])
#        data['contact'] = data['contact'].replace(['telephone','cellular'],[1,2])
#        data['poutcome'] = data['poutcome'].replace(['nonexistent','failure','success'],[1,2,3])
        
        return data
     
    
    
    def convertUnknownToNaN(self,data,pd,np):
        
        col_names = list(data)
        col_names_object = list(data.select_dtypes(include=['object']))
        for feat in col_names:
            data[feat] = data[feat].replace('unknown',np.nan)

#       Printing summary of the data for only categorical variables  
#        for feat in col_names_object:
#            f1 = data[feat]
#            print(feat)
#            print(pd.unique(f1))
#            print(f1.value_counts())
#            print()
            
        print("counts ratio before removing all the unknown values is",data['y'].value_counts()[1]/data['y'].value_counts()[0])
        
        return data
    
    
    
    def DropData_NAN(self,data,pd,np):
        
        return data.dropna()
    
    
    def OneHotVector(self,data,which_columns,pd,np):
        
        return pd.get_dummies(data, columns=which_columns)
    
    def ReplaceUnkClassWiseMeans(self,data,pd,np):
        
        class_yes_data1 = data[data['y'] == 'yes']
        class_no_data1 = data[data['y'] == 'no'] 
        values_no = {'job': np.round(class_no_data1['job'].mean()),'education': np.round(class_no_data1['education'].mean()), 'month': np.round(class_no_data1['month'].mean()), 'day_of_week': np.round(class_no_data1['day_of_week'].mean())}
        values_yes = {'job': np.round(class_yes_data1['job'].mean()),'education': np.round(class_yes_data1['education'].mean()), 'month': np.round(class_yes_data1['month'].mean()), 'day_of_week': np.round(class_yes_data1['day_of_week'].mean())}
        class_no_data1 = class_no_data1.fillna(value=values_no)
        class_yes_data1 = class_yes_data1.fillna(value=values_yes)
        
        frame = [class_no_data1,class_yes_data1]
        new_frame = pd.concat(frame)
        return new_frame.sort_index()
    
    def KNNcleaning(self,data,pd,np):
        
        from fancyimpute import KNN
        sel_columns = ['job','education','month','day_of_week','marital', 'housing', 'loan','contact','poutcome']
        data_useful_frame = data.loc[:,sel_columns]
        data_useful_matrix = np.array(data_useful_frame)
        modified_cols = pd.DataFrame(data=KNN(3).complete(data_useful_matrix), columns=data_useful_frame.columns, index=data_useful_frame.index)
        print(modified_cols.head())
        modified_cols = np.round(modified_cols)
        data[sel_columns] = modified_cols[sel_columns]
        
        data['marital'] = data['marital'].replace([1,2,3],['divorced','single','married'])
        data['housing'] = data['housing'].replace([1,2],['no','yes'])
        data['loan'] = data['loan'].replace([1,2],['no','yes'])
        data['contact'] = data['contact'].replace([1,2],['telephone','cellular'])
        data['poutcome'] = data['poutcome'].replace([1,2,3],['nonexistent','failure','success'])
        
        return data
    
    def StandardizeData(self,cleaned_mat,which_features,np):
        
        from sklearn import preprocessing
        selected_feat_matrix = cleaned_mat[:,0:len(which_features)]
        rest_feat_matrix = cleaned_mat[:,len(which_features):cleaned_mat.shape[1]]
        
        norm_feat_matrix = preprocessing.scale(selected_feat_matrix)
        X_scaled = np.concatenate((norm_feat_matrix,rest_feat_matrix),axis=1)
        return X_scaled
            
        

            

#------------------------------------------Replacing Unknown values with class Means---------------------------------------------#
data = pd.read_csv("bank-additional.csv")

ordered_vals_dict = {'job':['unemployed','student','housemaid','retired','self-employed','admin.','services','technician','blue-collar','management','entrepreneur'],
                     'education':['illiterate','basic.4y','basic.6y','basic.9y','high.school','professional.course','university.degree'],
                     'month':['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                     'day_of_week':['mon','tue','wed','thu','fri']
                    }

ordered_numeric_dict = {'job':[1,2,3,4,5,6,7,8,9,10,11],
                     'education':[1,2,3,4,5,6,7],
                     'month':[1,2,3,4,5,6,7,8,9,10],
                     'day_of_week':[1,2,3,4,5]
                    }

unordered_vals_dict = {'marital':['divorced','single','married'],
                       'housing':['no','yes'],
                       'loan':['no','yes'],
                       'contact':['telephone','cellular'],
                       'poutcome':['nonexistent','failure','success']
                      }

unordered_numeric_dict = {'marital':[1,2,3],
                       'housing':[1,2],
                       'loan':[1,2],
                       'contact':[1,2],
                       'poutcome':[1,2,3]
                      }



sel_cols_ordered = ['job','education','month','day_of_week']
sel_cols_unordered = ['marital', 'housing', 'loan','contact','poutcome']

process = PreProcessData()
#Remove default feature as it has 803 unknown values which is almost one fourth of the data set
data_NA = process.removeFeature(data,'default',pd,np)
col_names = list(data_NA.select_dtypes(include=['object']))
data_NA = process.AssignValsOrderedCategorical(data_NA,pd,np,ordered_vals_dict,ordered_numeric_dict)
data_NA = process.convertUnknownToNaN(data_NA,pd,np)
#If you want to consider data with all NAN values removed without any imputation, just remove the below line
data_NA_imputed = process.ReplaceUnkClassWiseMeans(data_NA,pd,np)
data_NA_unknown_removed = process.DropData_NAN(data_NA_imputed,pd,np)

process_KNN = PreProcessData()
#Remove default feature as it has 803 unknown values which is almost one fourth of the data_KNN set
data_KNN = process_KNN.removeFeature(data,'default',pd,np)
col_names = list(data_KNN.select_dtypes(include=['object']))
data_KNN = process_KNN.AssignValsOrderedCategorical(data_KNN,pd,np,ordered_vals_dict,ordered_numeric_dict)
data_KNN = process_KNN.AssignValsUnorderedCategorical(data_KNN,pd,np,unordered_vals_dict,unordered_numeric_dict)
data_KNN = process_KNN.convertUnknownToNaN(data_KNN,pd,np)
#If you want to consider data_KNN with all NAN values removed without any imputation, just remove the below line
data_KNN_imputed = process_KNN.KNNcleaning(data_KNN,pd,np)
data_KNN_unknown_removed = process_KNN.DropData_NAN(data_KNN_imputed,pd,np)

for feat in col_names:
    f1 = data_NA_unknown_removed[feat]
    
#    data[feat] = data[feat].astype('category')
    print(feat)
    print(pd.unique(f1))
    print(f1.value_counts())
    print()
    
for feat in col_names:
    f1 = data_KNN_unknown_removed[feat]
    
#    data_KNN[feat] = data_KNN[feat].astype('category')
    print(feat)
    print(pd.unique(f1))
    print(f1.value_counts())
    print()
    
#print("counts ratio after removing all the unknown values is",data_NA_unknown_removed['y'].value_counts()[1]/data_NA_unknown_removed['y'].value_counts()[0])
##As we can see the ratio of yes and no values(yes/no) is 0.12 before and after the removal of unknown values
# There removing data points will all unknown values might work in our case


## After doing this we need to convert the unordered categorical variables in one hot repreentation
data_NA_final = process.OneHotVector(data_NA_unknown_removed,sel_cols_unordered,pd,np)
data_KNN_final = process_KNN.OneHotVector(data_KNN_unknown_removed,sel_cols_unordered,pd,np)

#Change 999 of pddays to -1 so that it is more meaningful
data_NA_final['pdays'] = data_NA_final['pdays'].replace(999,-1)
#Change the class labels to 0 and 1 only
data_NA_final['y'] = data_NA_final['y'].replace(['yes','no'],[1,0])

#Change 999 of pddays to -1 so that it is more meaningful
data_KNN_final['pdays'] = data_KNN_final['pdays'].replace(999,-1)
#Change the class labels to 0 and 1 only
data_KNN_final['y'] = data_KNN_final['y'].replace(['yes','no'],[1,0])

######Taking the class labels column to the end of the data##############
col_names = list(data_NA_final)
class_label_name = col_names.pop(col_names.index('y'))
col_names.insert(data_NA_final.shape[1]-1,class_label_name)
data_NA_final = data_NA_final.loc[:,col_names]

######Taking the class labels column to the end of the data_KNN##############
col_names = list(data_KNN_final)
class_label_name = col_names.pop(col_names.index('y'))
col_names.insert(data_KNN_final.shape[1]-1,class_label_name)
data_KNN_final = data_KNN_final.loc[:,col_names]

#####Standardize data##################################################
features_to_standardize = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
cleaned_data_NA = np.array(data_NA_final)
standardized_cleaned_data_NA = process.StandardizeData(cleaned_data_NA,features_to_standardize,np)


#####Standardize data_KNN##################################################
features_to_standardize = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
cleaned_data_KNN = np.array(data_KNN_final)
standardized_cleaned_data_KNN = process_KNN.StandardizeData(cleaned_data_KNN,features_to_standardize,np)


X_NA_removed = standardized_cleaned_data_NA[:,0:(standardized_cleaned_data_NA.shape[1]-1)]
y_NA_removed = standardized_cleaned_data_NA[:,(standardized_cleaned_data_NA.shape[1]-1):standardized_cleaned_data_NA.shape[1]]
y_NA_removed = np.reshape(y_NA_removed,(y_NA_removed.shape[0],))

X_KNN = standardized_cleaned_data_KNN[:,0:(standardized_cleaned_data_KNN.shape[1]-1)]
y_KNN = standardized_cleaned_data_KNN[:,(standardized_cleaned_data_KNN.shape[1]-1):standardized_cleaned_data_KNN.shape[1]]
y_KNN = np.reshape(y_KNN,(y_KNN.shape[0],))


#----------------------------------The BaseLine Model------------------------------------------

process_baseline = PreProcessData()

data_baseline = process_baseline.removeFeature(data,'default',pd,np)
col_names = list(data_baseline.select_dtypes(include=['object']))
data_baseline = process.convertUnknownToNaN(data_baseline,pd,np)
data_baseline_unknown_removed = process.DropData_NAN(data_baseline,pd,np)


for feat in col_names:
    f1 = data_baseline_unknown_removed[feat]
    
#    data_KNN[feat] = data_KNN[feat].astype('category')
    print(feat)
    print(pd.unique(f1))
    print(f1.value_counts())
    print()

data_baseline_final = process_baseline.OneHotVector(data_baseline_unknown_removed,sel_cols_ordered+sel_cols_unordered,pd,np)
data_baseline_final['y'] = data_baseline_final['y'].replace(['yes','no'],[1,0])

######Taking the class labels column to the end of the data_KNN##############
col_names = list(data_baseline_final)
class_label_name = col_names.pop(col_names.index('y'))
col_names.insert(data_baseline_final.shape[1]-1,class_label_name)
data_baseline_final = data_baseline_final.loc[:,col_names]

data_baseline_final = np.array(data_baseline_final)
X_baseline = data_baseline_final[:,0:(data_baseline_final.shape[1]-1)]
y_baseline = data_baseline_final[:,(data_baseline_final.shape[1]-1):data_baseline_final.shape[1]]

#Splitting the train and the test set

s1 = 'ROC curve with AUC for Baseline model'
plt_roc.plotRocCurve_Unbalanced(X_baseline,y_baseline,2,s1)

############Feature selection K best features using RFE####################


s1 = "25 ROC curves with 25 to 1 feature selected every iteration using Linear Perceptron for X_NA_removed"
s2 = "25 bar plots with 25 to 1 feature selected every iteration using Linear Perceptron for X_NA_removed"
feat_f1_NA_removed_percep = p_CV.CVFeatureSel_Percep(X_NA_removed,y_NA_removed,s1,s2)

s1 = "25 ROC curves with 25 to 1 feature selected every iteration using Linear Perceptron for X_KNN"
s2 = "25 bar plots with 25 to 1 feature selected every iteration using Linear Perceptron for X_KNN"
feat_f1_KNN_percep = p_CV.CVFeatureSel_Percep(X_KNN,y_KNN,s1,s2)

s1 = "25 ROC curves with 25 to 1 feature selected every iteration using SVM rbf kernel for X_NA_removed"
s2 = "25 bar plots with 25 to 1 feature selected every iteration using SVM rbf kernel for X_NA_removed"
feat_f1_NA_removed = p_CV.CVFeatureSel_SVM(X_NA_removed,y_NA_removed,s1,s2)
feat_size_NA = []
f1_score_NA = []
for keys in feat_f1_NA_removed:
    feat_size_NA.append(keys[0])
    f1_score_NA.append(keys[1])
idx = np.argmax(f1_score_NA)
k = (feat_size_NA[idx],f1_score_NA[idx])
selector_NA_KNN = feat_f1_NA_removed.get(k)
X_NA_removed = X_NA_removed[:,list(selector_NA_KNN)]


s1 = "25 ROC curves with 25 to 1 feature selected every iteration using SVM rbf kernel for X_KNN"
s2 = "25 bar plots with 25 to 1 feature selected every iteration using SVM rbf kernel for X_KNN"
feat_f1_KNN = p_CV.CVFeatureSel_SVM(X_KNN,y_KNN,s1,s2)
X_KNN = X_KNN[:,list(selector_NA_KNN)]


############----------------------perceptron---------------------##############

X_NA_removed_dim_red = X_NA_removed
X_KNN_dim_red = X_KNN

dim_estimators_NA = p_CV.CVPerceptronfinal(X_NA_removed,y_NA_removed)
s1 = 'ROC curve with AUC for NA values removed for all unordered categorical variable UNBALANCED:'
plt_roc.plotRocCurve_Unbalanced(X_NA_removed_dim_red,y_NA_removed,dim_estimators_NA.get('best_weight'),s1,dim_estimators_NA.get('best_max_iter'))

dim_estimators_KNN = p_CV.CVPerceptronfinal(X_KNN,y_KNN)
s1 = 'ROC curve with AUC for KNN imputation UNBALANCED:'
plt_roc.plotRocCurve_Unbalanced(X_KNN_dim_red,y_KNN,dim_estimators_KNN.get('best_weight'),s1,dim_estimators_KNN.get('best_max_iter'))

dim_estimators_NA_balanced = p_CV.CVPerceptronfinal_balanced(X_NA_removed,y_NA_removed)
s1 = 'ROC curve with AUC for NA values removed for all unordered categorical variable BALANCED:'
plt_roc.plotRocCurve_balanced(X_NA_removed_dim_red,y_NA_removed,s1,dim_estimators_NA_balanced.get('best_max_iter'))

dim_estimators_KNN_balanced = p_CV.CVPerceptronfinal_balanced(X_KNN,y_KNN)
s1 = 'ROC curve with AUC for KNN imputation BALANCED:'
plt_roc.plotRocCurve_balanced(X_KNN_dim_red,y_KNN,s1,dim_estimators_KNN_balanced.get('best_max_iter'))


############------------------SVM-----------------------------#####################

s1 = 'ROC curve with AUC for NA values removed for all unordered categorical variable UNBALANCED using SVM rbf Kernel:'
p_CV.CVSVM_Unbalanced(X_NA_removed,y_NA_removed,s1,9)

s1 = 'ROC curve with AUC for KNN imputed data UNBALANCED using SVM rbf Kernel:'
p_CV.CVSVM_Unbalanced(X_KNN,y_KNN,s1,9)

s1 = 'ROC curve with AUC for NA values removed for all unordered categorical variable BALANCED using SVM rbf Kernel:'
p_CV.CVSVM_balanced(X_NA_removed,y_NA_removed,s1,9)

s1 = 'ROC curve with AUC for KNN imputed data BALANCED using SVM rbf Kernel:'
p_CV.CVSVM_balanced(X_KNN,y_KNN,s1,9)



###############Bayes Minimum risk##################################################

###########################Feature selection for Bayes Minimum risk

s1 = "ROC curves for all features from 1 to 25 X_NA_removed"
s2 = "F1 scores and ROC for Min risk Bayes for features ranging from 1 to 25 X_NA_removed"
feat_f1_NA_removed = p_CV.CVFeature_sel_Bayes(X_NA_removed,y_NA_removed,s1,s2)
feat_size_NA = []
f1_score_NA = []
for keys in feat_f1_NA_removed:
    feat_size_NA.append(keys[0])
    f1_score_NA.append(keys[1])
idx = np.argmax(f1_score_NA)
k = (feat_size_NA[idx],f1_score_NA[idx])
selector_NA = feat_f1_NA_removed.get(k)
X_NA_removed = X_NA_removed[:,list(selector_NA)]

s1 = "ROC curves for all features from 1 to 25 X_KNN"
s2 = "F1 scores for Min risk Bayes for features ranging from 1 to 25 X_KNN"
feat_f1_KNN = p_CV.CVFeature_sel_Bayes(X_KNN,y_KNN,s1,s2)

feat_size_KNN = []
f1_score_KNN = []
for keys in feat_f1_KNN:
    feat_size_KNN.append(keys[0])
    f1_score_KNN.append(keys[1])
idx = np.argmax(f1_score_KNN)
k = (feat_size_KNN[idx],f1_score_KNN[idx])
selector_KNN = feat_f1_KNN.get(k)
X_KNN = X_KNN[:,list(selector_KNN)]


######################Minimumrisk classifier##########################

X_err_train, X_err_test, y_err_train, y_err_test =  train_test_split(X_NA_removed,y_NA_removed, test_size=0.1, random_state=42,stratify = y_NA_removed)



f1_scorer = make_scorer(f1_score,pos_label=1)

bandwidths = np.arange(1,4,0.1)
L10 = np.arange(1,10,1)
L01 = np.arange(1,10,1)



k = ['gaussian','tophat','epanechnikov','exponential','linear','cosine']


grid = GridSearchCV(KernelDensityClassifier(),{'bandwidth': bandwidths,'Lambda10':L10,'Lambda01':L01},scoring=f1_scorer,n_jobs=-1)
grid.fit(X_err_train,y_err_train)


bw = grid.best_params_.get('bandwidth')
Lam10 = grid.best_params_.get('Lambda10')
Lam01 = grid.best_params_.get('Lambda01')



s1 = "ROC curve for Bayes Minimum error X_NA_removed"
plt_roc.plotRoc_Bayes(X_NA_removed,y_NA_removed,bw,'gaussian',Lam10,Lam01,s1)



X_err_train, X_err_test, y_err_train, y_err_test =  train_test_split(X_KNN,y_KNN, test_size=0.1, random_state=42,stratify = y_KNN)

grid1 = GridSearchCV(KernelDensityClassifier(),{'bandwidth': bandwidths,'Lambda10':L10,'Lambda01':L01},scoring=f1_scorer,n_jobs=-1)
grid1.fit(X_err_train,y_err_train)


bw = grid1.best_params_.get('bandwidth')
Lam10 = grid1.best_params_.get('Lambda10')
Lam01 = grid1.best_params_.get('Lambda01')


############################Neural network############################



s1 = 'Neural net CV results X_NA_removed'
p_CV.CVNeuralnet(X_NA_removed,y_NA_removed,s1)

s1 = 'Neural net CV results X_KNN'
p_CV.CVNeuralnet(X_KNN,y_KNN,s1)
