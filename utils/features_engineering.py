import pandas as pd
import numpy as np

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
