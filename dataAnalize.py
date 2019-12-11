# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:41:37 2019

@author: tiago
"""

from pcaFS import pca_full_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas
import gc
import time

data = pandas.read_hdf("training_features.h5", key = 'features')

def personCorrelationList(data):
    correlation = []
    for index,coluna1 in enumerate(data.columns):
        for coluna2 in data.columns[index+1:]:
            correlation.append((pearsonr(data[coluna1],data[coluna2])[0],coluna1,coluna2))
    return sorted(correlation,key = lambda x:abs(x[0]),reverse=True)

def testModel(model, data, target, NofFolds):
    kf = KFold(n_splits=NofFolds)
    kf.get_n_splits(data)
    accuracies = []
    for train_index, test_index in kf.split(data):
        train_x, test_x = data.iloc[train_index], data.iloc[test_index]
        train_y, test_y = target.iloc[train_index], target.iloc[test_index]
        model.fit(train_x, train_y)
        score = model.score(test_x, test_y)
        accuracies.append(score)
    #train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2)
    return np.mean(accuracies)

def runClassificationTests(data, target):
    models = {}
    models['RF'] = RandomForestClassifier(n_estimators = 150)
    models['SVM'] = SVC(kernel='linear', C = 0.025,max_iter = 800, verbose = False)
    mlpParams = {
                'hidden_layer_sizes':(5,5),
                'activation':'logistic',
                'solver' : 'adam',
                'learning_rate_init' : 0.001,
                'max_iter': 500,
                'verbose': False
                }
    models['MLP'] = MLPClassifier(**mlpParams)
    models['KNN'] = KNeighborsClassifier(n_neighbors = 20)
    
    result = {}
    for modelName in models.keys():
        #result[modelName] = testModel(models[modelName], data, target, 10)
        print("accuray of ", modelName, " is " , testModel(models[modelName], data, target, 10))
    return result

def pcaFeatureSelection(data,k):
    _,_,_,_,_,_,_,featureRanking = pca_full_report(data.drop(columns='target').to_numpy(), features_=list(data.drop(columns='target').columns),save_plot=False)
    first15 = featureRanking.head(k)
    return data[list(first15.feature_)  + ["target"]]

def imputeData(data):
    imputer = SimpleImputer(strategy = 'mean')
    imputedData = pandas.DataFrame(imputer.fit_transform(data),columns = data.columns, index = data.index)
    return imputedData
    
if __name__ == '__main__':
    results = []
    resultsIndex = []
    #for k in reversed(range(19,110,10)):
    #read data and create dataFrame
    data = pandas.read_hdf("FeaturesPassbands.h5", key = 'features')
    data.index = data.object_id
    data.drop(columns='object_id',inplace = True)
        
    #impute Nan values
    data = imputeData(data)
        
    #select features with pca, change next line to false to not use it
    usePcaFeatureSelection = False
    if(usePcaFeatureSelection):
        data = pcaFeatureSelection(data, k)
        
    #divide features from target
    x = data.drop(columns = 'target')
    y = data.target
        
    #select features with sklearn methods, change next line to false to not use it
    # methods tested: chi2, f_classif, mutual_info_classif
    useSklearnFeatureSelection = False
    if(useSklearnFeatureSelection):
        selector = SelectKBest(f_classif, k)
        selector.fit(x,y)
            
        cols = selector.get_support(indices = True)
        x = x.iloc[:,cols]
            
        
    #normilize database using z-score
    stdScaler = StandardScaler(copy = False)
    stdScaler.fit_transform(x)
            
    #Create new features with PCA, change next line to false to not use it
    #needs to be done after normalization
    usePCATransformation = False
    if(usePCATransformation):
        pca_ = PCA().fit(x)
        x = PCA().fit_transform(x)
        x = pandas.DataFrame(x)
        
    #tests the classification models and print accuracy
    result = runClassificationTests(x,y)
    #print(k , result)
    #results.append(result)
    #resultsIndex.append(k)
        
    #resultsDF = pandas.DataFrame(results, index = resultsIndex)
    
    
    
#corList = personCorrelationList(data)