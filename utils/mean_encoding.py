import pandas as pd
import numpy as np

def mean_encoding(data_x, data_y, feature, target):
    data = pd.concat([data_x, data_y], axis=1)
    prior_mean = np.mean(data[target].values)
    enc_value = data.groupby(feature)[target].mean()
    if isinstance(feature, list):
        feature_name = 'enc_' + '_'.join(feature)
        data_x[feature_name] = data_x[f].join(pd.DataFrame(enc_value), on=feature, how='left')[target]
    else:
        feature_name = 'enc_' + feature
        data_x[feature_name] = data_x[feature].map(enc_value)    
    return data_x, enc_value, prior_mean, feature_name

def smooth_mean_encoding(data_x, data_y, feature, target, min_samples=1, smooth_method='stats_smooth'):
    data = pd.concat([data_x, data_y], axis=1)
    nrows = data.groupby(feature)[target].count()
    target_mean = data.groupby(feature)[target].mean()
    prior_mean = np.mean(data[target].values)    
    if smooth_method == 'stats_smooth':
        smooth_enc_value = (target_mean * nrows + prior_mean * min_samples) / (nrows + min_samples) 
    elif smooth_method == 'sigmoid_smooth':
        smoothing_slope = 1
        smooth_factor = 1 / (1 + np.exp(- (nrows - min_samples) / smoothing_slope))
        smooth_enc_value = smooth_factor * target_mean + (1 - smooth_factor) * prior_mean        
    if isinstance(feature, list):
        feature_name = 'smooth_enc_' + '_'.join(feature)
        data_x[feature_name] = data_x[f].join(pd.DataFrame(smooth_enc_value), on=feature, how='left')[target]
    else:
        feature_name = 'smooth_enc_' + feature
        data_x[feature_name] = data_x[feature].map(smooth_enc_value)          
    return data_x, smooth_enc_value, prior_mean, feature_name

def beta_mean_encoding(data, feature, target, stats, prior_mean, N_min=5):
    df_stats = pd.merge(data[[feature]], stats, how='left', on=feature)
    df_stats['sum'].fillna(value=prior_mean, inplace = True)
    df_stats['count'].fillna(value=1.0, inplace = True)    
    N_prior = np.maximum(N_min - df_stats['count'].values, 0)   # prior parameters
    df_stats[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (N_prior + df_stats['count']) # Bayesian mean
    
    if isinstance(feature, list):
        feature_name = 'mec_' + '_'.join(feature)
        data[feature_name] = df_stats[feature_name].values
    else:
        feature_name = 'mec_' + feature
        data[feature_name] = df_stats[feature_name].values         
    return data, feature_name
