import pandas as pd
import lightgbm as lgb
import traceback
import os
import multiprocessing
import xgboost as xgb
import subprocess
import numpy as np
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from tensorflow.python.lib.io import file_io
import tensorflow as tf

# global setting
number_of_iter = 1000
early_stopping = 30
seed = 202109


def Optimization(model_name, train_df, validation_df, target, predictors, categorical):
    if model_name == 'lightgbm':
        bayes_cv_tuner = BayesSearchCV(
            estimator = lgb.LGBMClassifier(objective='binary', metric='auc', n_jobs=1, verbose=0),
            search_spaces = {'learning_rate': (0.01, 1.0, 'log-uniform'),
                             'num_leaves': (1, 100),
                             'max_depth': (0, 50),
                             'min_child_samples': (0, 50),
                             'max_bin': (100, 1000),
                             'subsample': (0.01, 1.0, 'uniform'),
                             'subsample_freq': (0, 10),
                             'colsample_bytree': (0.01, 1.0, 'uniform'),
                             'min_child_weight': (0, 10),
                             'subsample_for_bin': (100000, 500000),
                             'reg_lambda': (1e-9, 1000, 'log-uniform'),
                             'reg_alpha': (1e-9, 1.0, 'log-uniform'),
                             'scale_pos_weight': (1e-6, 500, 'log-uniform')},
            scoring = 'roc_auc',
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
            n_jobs = 1,
            n_iter = number_of_iter,
            verbose = 0,
            refit = True,
            random_state = seed)
        # Fit the model
        def status_print(optim_result):
            """Status callback durring bayesian hyperparameter search"""
            # Get all the models tested so far in DataFrame format
            all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

            # Get current parameters and the best parameters
            best_params = pd.Series(bayes_cv_tuner.best_params_)
            print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(len(all_models),
                                                                          np.round(bayes_cv_tuner.best_score_, 4),
                                                                          bayes_cv_tuner.best_params_))

            # Save all model results
            clf_name = bayes_cv_tuner.estimator.__class__.__name__
            output_path = os.path.join('/tmp', (clf_name + "_cv_results.csv"))
            all_models.to_csv(output_path)
            cmd = 'gsutil cp {} {}'.format(output_path, 'gs://onead-gcpml/ninja_project/output')
            res = subprocess.check_call(cmd, shell=True)

        result = bayes_cv_tuner.fit(train_df[predictors].values, train_df[target].values, callback=status_print)
        print(result, result.__dir__)

    elif model_name == 'xgboost':
        pass


def lgbm_model(train_df, validation_df, target, predictors, categorical):
    set_params = {
        'learning_rate': 0.04,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 31,  # 2^max_depth - 1
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200, # because training data is extremely unbalanced
        'reg_alpha': 0.99,
        'reg_lambda': 0.9
    }
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': multiprocessing.cpu_count()-1,
        'verbose': 0}
    params.update(set_params)

    print('[INFO] Training lightGBM model...')
    try:
        x_train = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                              feature_name=predictors, categorical_feature=categorical)
        x_validation = lgb.Dataset(validation_df[predictors].values, label=validation_df[target].values,
                                   feature_name=predictors, categorical_feature=categorical)
        eval_results = {}
        model = lgb.train(params, x_train, valid_sets=[x_validation], valid_names=['valid'],
                          evals_result=eval_results, num_boost_round=number_of_iter,
                          early_stopping_rounds=early_stopping, verbose_eval=10, feval=None)
        n_estimators = model.best_iteration
        print("\nModel Report:")
        print("n_estimators : ", n_estimators)
        print("AUC :", eval_results['valid']['auc'][n_estimators-1])
        return model, n_estimators
    except Exception as e:
        print(e, traceback.print_exc())


def xgboost_model(train_df, validation_df, target, predictors):
    set_params = {
        'eta': 0.3,
        'max_depth': 6,
        'learning_rate': 0.3,
        'n_estimators': 500, # number of trees, usually range between 50-1000
        'nthread':24,
        'njobs': -1,
        'gamma': 5.103973694670875e-08,
        'max_delta_step': 20,
        'min_child_weight': 4,
        'subsample': 0.7,
        'colsample_bylevel': 0.1,
        'colsample_bytree': 0.7,
        'reg_alpha': 1e-09,
        'reg_lambda': 1000.0,
        'scale_pos_weight': 499.99999999999994,
        'tree_method':'approx'
    }

    params = {
        'eta': 0.1,
        'tree_method': "hist", # Fast histogram optimized approximate greedy algorithm.
        'grow_policy': "lossguide", # split at nodes with highest loss change
        'max_leaves': 1400, # Maximum number of nodes to be added. (for lossguide grow policy)
        'max_depth': 0, # 0 means no limit (useful only for depth wise grow policy)
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.1,
        'min_child_weight': 0, # The larger, the more conservative the algorithm will be
        'alpha': 4, # L1 regularization. on weights | large value = more conservative model (maintain more features)
        'gamma': 5.103973694670875e-08,
        'objective': 'binary:logistic',
        'scale_pos_weight': 90,
        'eval_metric': 'auc',
        'nthread': multiprocessing.cpu_count()-1,
        'random_state': 202109,
        'silent': False}
    params.update(set_params)

    gpu_params = {
        'eta': 0.1,
        'tree_method': "gpu_hist", # Fast histogram optimized approximate greedy algorithm.
        'grow_policy': "lossguide", # split at nodes with highest loss change
        'max_leaves': 1400, # Maximum number of nodes to be added. (for lossguide grow policy)
        'max_depth': 0, # 0 means no limit (useful only for depth wise grow policy)
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'colsample_bylevel': 0.1,
        'min_child_weight': 0, # The larger, the more conservative the algorithm will be
        'alpha': 4, # L1 regularization. on weights | large value = more conservative model (maintain more features)
        'gamma': 0,
        'objective': 'binary:logistic',
        'scale_pos_weight': 90,
        'eval_metric': 'auc',
        'nthread': multiprocessing.cpu_count()-1,
        'random_state': 202109,
        'silent': False
    }
    print('[INFO] Training XGBoost model...')
    try:
        dtrain = xgb.DMatrix(train_df[predictors], train_df[target])
        dvalid = xgb.DMatrix(validation_df[predictors], validation_df[target])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        xgb_model = xgb.train(params=gpu_params, dtrain=dtrain, num_boost_round=number_of_iter, evals=watchlist,
                              early_stopping_rounds = early_stopping, verbose_eval=10)
    except Exception as e:
        print(e, traceback.print_exc())
    return xgb_model

def catboost_model(train_df, validation_df, target, predictors, categorical_index):
    params = {
      'iterations': number_of_iter, # alias n_estimators
      'learning_rate' : 0.1, # alias eta
      'depth': 6, # alias max_depth
      'loss_function': 'Logloss',
      'l2_leaf_reg': 5, # L2 regularization coefficient. Used for leaf value calculation.
      # 'subsample': 0.7, # Can't used when bootstrap_type = 'Bayesian'(default).
      'colsample_bylevel': 0.7,
      'scale_pos_weight': 99.7,
      'eval_metric': 'AUC',
      'random_state': 202109,
      'calc_feature_importance': True,
      'thread_count': multiprocessing.cpu_count()-1,
      'verbose': True,
      'one_hot_max_size': 24 # Use one-hot encoding for all features with number of different values less than or equal to the given parameter value
      # 'ignored_features'
    }
    catboost_model = CatBoostClassifier(**params, od_type='Iter', od_wait=early_stopping)
    catboost_model.fit(train_df[predictors], train_df[target], cat_features=categorical_index,
                   eval_set=(validation_df[predictors], validation_df[target]),
                   use_best_model=True, verbose=True)
    return catboost_model
