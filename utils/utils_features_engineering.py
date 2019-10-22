import pandas as pd
import numpy as np
import gc

def do_count(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Count {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].count()\
        .reset_index()\
        .rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data
    
def do_count_unique(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Count unique {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].nunique()\
        .reset_index()\
        .rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def do_cumcount(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Cumcount {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].cumcount()
    data[new_col_name] = group_data.values
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data
    
def do_mean(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Compute mean {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].mean()\
        .reset_index().rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def do_var(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Compute var {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].var()\
        .reset_index()\
        .rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def do_sum(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Compute mean {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].sum()\
        .reset_index().rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def do_max(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Compute mean {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].max()\
        .reset_index().rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def do_min(data, group_cols, target_col, new_col_name, col_type):
    print('[INFO] Compute mean {} with group by {} combination...'.format(target_col, '-'.join(group_cols)))
    cols = group_cols.copy()
    cols.append(target_col)
    group_data = data[cols].groupby(by=group_cols)[[target_col]].min()\
        .reset_index().rename(index=str, columns={target_col: new_col_name})
    data = data.merge(group_data, on=group_cols, how='left')
    del group_data
    data[new_col_name] = data[new_col_name].astype(col_type)
    gc.collect()
    return data

def mean_encoding(data_x, data_y, feature, target):
    if target in data_x.columns:
        data = data_x
    else:
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
    if target in data_x.columns:
        data = data_x
    else:
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

