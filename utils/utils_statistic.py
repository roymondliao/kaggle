import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def resumetable(df):
    print("Dataset Shape: {}".format(df.shape))
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Missing%'] = np.around(df.isnull().mean().values * 100, 4)
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

    
def covariate_shift(train_data, test_data, feature, params=None, seed=42):
    default_params = {
        'objective': 'binary', 
        'boosting_type': 'gbdt', 
        'subsample': 1, 
        'bagging_seed': 11, 
        'metric': ['auc'], 
        'random_state': 42
    }
    if params is not None:
        default_params.update(params)
    
    df_train = pd.DataFrame(data={feature: train_data[feature], 'is_test': 0})
    df_test = pd.DataFrame(data={feature: test_data[feature], 'is_test': 1})

    # Creating a single dataframe
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
    
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['is_test'], 
                                                        test_size=0.33, random_state=seed, stratify=df['is_test'])
    clf = lgb.LGBMClassifier(**default_params, num_boost_round=500)
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])

    del df, X_train, y_train, X_test, y_test
    gc.collect()
    
    return roc_auc