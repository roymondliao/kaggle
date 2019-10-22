#!/usr/bin/env python3

from multiprocessing import Pool
import multiprocessing
import warnings
import os
import gc
import random
import itertools
import pickle
from pathlib import Path
from collections import Counter
from datetime import datetime

# data preprocessing 
from itertools import product
import pandas as pd
import numpy as np
import missingno
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

# model
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin

# eveluation 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, KFold


warnings.filterwarnings('ignore')
seed = 9527



class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    '''
        This class should implement KNN features extraction 
    '''
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric
        
        if n_neighbors is None:
            self.n_neighbors = max(k_list) 
        else:
            self.n_neighbors = n_neighbors
            
        self.eps = eps        
        self.n_classes_ = n_classes
    
    def fit(self, X, y):
        '''
            Set's up the train set and self.NN object
        '''
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                   metric=self.metric, 
                                   n_jobs=self.n_jobs, 
                                   algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)
        self.y_train = y
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_                
        
    def predict(self, X):       
        '''
            Produces KNN features for every object of a dataset X
        '''
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):               
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            feat = []
            with Pool(processes=self.n_jobs) as pool:
                res = pool.map(self.get_features_for_one, list(X))     
                #multi_res = [pool.apply_async(self.get_features_for_one, (np.array(i),)) for i in X]
                #for res in multi_res:
                #    feat.append(res.get())
        return np.vstack(res)
        
        
    def get_features_for_one(self, x):
        '''
            Computes KNN features for a single object `x`
        '''       
        NN_output = self.NN.kneighbors(x.reshape(1, -1))
        
        # Stores indices of the neighbors
        neighs = NN_output[1][0]
    
        # Stores distances to corresponding neighbors
        neighs_dist = NN_output[0][0] 
        
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[neighs] 
        
        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = [] 
        
        
        ''' 
            1. Fraction of objects of every class.
               It is basically a KNNСlassifiers predictions.

               Take a look at `np.bincount` function, it can be very helpful
               Note that the values should sum up to one
        '''
        for k in self.k_list:
            feats = np.bincount(neighs_y[:k], minlength=self.n_classes)
            feats = feats / sum(feats)
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        
        '''
            2. Same label streak: the largest number N, 
               such that N nearest neighbors have the same label.
               
               What can help you: `np.where`
        '''
        diff = np.where(neighs_y != neighs_y[0])[0]
        feats = [diff[0]] if len(diff) else [len(neighs_y)]
        
        assert len(feats) == 1
        return_list += [feats]        

        
        '''
            3. Minimum distance to objects of each class
               Find the first instance of a class and take its distance as features.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.

               `np.where` might be helpful
        '''
        feats = []
        for c in range(self.n_classes):        
            first_instance = np.where(neighs_y == c)[0]
            feats.append(neighs_dist[first_instance[0]] if len(first_instance) else 999)
        
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            4. Minimum *normalized* distance to objects of each class
               As 3. but we normalize (divide) the distances
               by the distance to the closest neighbor.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.
               
               Do not forget to add self.eps to denominator.
        '''
        feats = []
        for c in range(self.n_classes):
            # neights_dist 是按照距離小排到大的數值
            same = np.where(neighs_y == c)[0]
            # normalized 都除以 minimum
            feats.append(neighs_dist[same[0]] / (self.eps + neighs_dist[0]) if len(same) else 999)                    
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            5. 
               5.1 Distance to Kth neighbor
                   Think of this as of quantiles of a distribution
               5.2 Distance to Kth neighbor normalized by 
                   distance to the first neighbor
               
               feat_51, feat_52 are answers to 5.1. and 5.2.
               should be scalars
               
               Do not forget to add self.eps to denominator.
        '''
        for k in self.k_list:            
            feat_51 = neighs_dist[k - 1] 
            feat_52 = neighs_dist[k - 1] / (self.eps + neighs_dist[0])            
            return_list += [[feat_51, feat_52]]
        
        '''
            6. Mean distance to neighbors of each class for each K from `k_list` 
                   For each class select the neighbors of that class among K nearest neighbors 
                   and compute the average distance to those objects
                   
                   If there are no objects of a certain class among K neighbors, set mean distance to 999
                   
               You can use `np.bincount` with appropriate weights
               Don't forget, that if you divide by something, 
               You need to add `self.eps` to denominator.
        '''
        for k in self.k_list:
            numerator = np.bincount(neighs_y[:k], weights=neighs_dist[:k], minlength=self.n_classes)
            denominator = self.eps + np.bincount(neighs_y[:k], minlength=self.n_classes)
            feats = np.where(numerator > 0, numerator / denominator, 999)     
            
            assert len(feats) == self.n_classes
            return_list += [feats]

        # merge        
        knn_feats = np.round(np.hstack(return_list), 6)
        return knn_feats

if __name__ == '__main__':
    main_path = Path('../input/ieee-cis-fraud-detection/')
    
    with open(str(main_path / 'train_df.pkl'), 'rb') as handle:
        train_df = pickle.load(handle)
        
    with open(str(main_path / 'scale_train_df.pkl'), 'rb') as handle:
        tmp_train_df = pickle.load(handle)

    with open(str(main_path / 'scale_test_df.pkl'), 'rb') as handle:
        tmp_test_df = pickle.load(handle)    
    
    target_col = 'isFraud'
    redundant_cols = ['P_emaildomain', 'R_emaildomain', 'DeviceType', 'DeviceInfo', 'TransactionDT', 'tran_date']
    features_cols = [c for c in tmp_train_df.columns.tolist() if c not in redundant_cols + ['TransactionID', 'isFraud']]
    
    tmp_train_df[target_col] = train_df[target_col]
    sample_train_df = tmp_train_df[tmp_train_df[target_col] == 0].sample(train_df[train_df.isFraud == 1].shape[0]*2,
                                                                         replace=False)
    sample_train_df = pd.concat([sample_train_df, tmp_train_df[tmp_train_df[target_col] == 1]], axis=0)
    k_list = [5, 9, 23]
    knn_feates_train = np.zeros((tmp_train_df.shape[0], max(k_list)))
    knn_feates_test = np.zeros((tmp_test_df.shape[0], max(k_list)))
    n_splits = 3

    for metric in ['minkowski', 'cosine']:
        print (metric)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)    
        #skf = TimeSeriesSplit(n_splits=n_splits)
        Kflods = skf.split(sample_train_df[features_cols].values, sample_train_df[target_col].values)

        for fold, (trn_idx, val_idx) in enumerate(Kflods):
            print("===== Fold {} =====".format(fold))
            trn_idx = shuffle(trn_idx)
            train_x, train_y = sample_train_df.iloc[trn_idx][features_cols], sample_train_df.iloc[trn_idx][target_col]
            valid_x, valid_y = sample_train_df.iloc[val_idx][features_cols], sample_train_df.iloc[val_idx][target_col]

            # Create instance of our KNN feature extractor
            NNF = NearestNeighborsFeats(n_jobs=multiprocessing.cpu_count(), k_list=k_list, metric=metric)
            NNF.fit(train_x.values, train_y.values)
            knn_feates_train += NNF.predict(tmp_train_df[features_cols].values) / skf.n_splits
            knn_feates_test += NNF.predict(tmp_test_df[features_cols].values) / skf.n_splits
        np.save(str(main_path / 'knn_feats_{}_train.npy'.format(metric)), knn_feates_train)
        np.save(str(main_path / 'knn_feats_{}_test.npy'.format(metric)), knn_feates_test)
