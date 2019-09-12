import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gc
from sklearn.preprocessing import LabelEncoder

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_feature_distribution(df1, df2, label1, label2, features, bins=10, color1='coral', color2='mediumseagreen'):
    assert isinstance(features, list), 'Features should be list type'        
    sns.set_style('darkgrid')
    n_features = len(features)
        
    if n_features > 4:
        rows = int(np.ceil(n_features / 4))
        cols = 4 - 1
    else:
        rows = 1
        cols = n_features - 1
    if n_features >= 4*7:
        plt.figure(figsize=(24, 22))
    else:
        plt.figure(figsize=(16, 14))
    i = 0
    for feature in features:
        i += 1
        df1_tmp = df1[feature][~df1[feature].isna()]
        df2_tmp = df2[feature][~df2[feature].isna()]
        if df1[feature].dtype == 'O':
            enc = LabelEncoder()
            enc.fit(df1_tmp.astype(str))
            df1_tmp = enc.transform(df1_tmp)
            df2_tmp = enc.transform(df2_tmp)
        plt.subplot(rows, cols + 1, i)        
        sns.distplot(df1_tmp, label=label1, kde=True, rug=False, hist=True, bins=bins, color=color1) 
        sns.distplot(df2_tmp, label=label2, kde=True, rug=False, hist=True, bins=bins, color=color2) 
        plt.xlabel(feature, fontsize=9)    
        plt.legend(labels=[label1, label2])
        locs, labels = plt.xticks()        
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-5)
        plt.tick_params(axis='y', which='major', labelsize=10)
        plt.subplots_adjust(hspace=0.25, top=0.85, wspace=0.35)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
    plt.show()
    del df1, df2
    gc.collect()

def plot_feature_distribution_old(df1, df2, label1, label2, features):
    assert isinstance(features, list), 'Features should be list type'        
    sns.set_style('darkgrid')
    n_features = len(features)    
    if n_features > 4:
        rows = int(np.ceil(n_features / 4))
        cols = 4
        fig, ax = plt.subplots(rows, cols, figsize=(16, 14))   
        ax = ax.flatten()
    else:
        rows = 1
        cols = n_features                        
        fig, ax = plt.subplots(rows, cols, figsize=(10, 8))    
    for i, feature in enumerate(features):
        df1_tmp = df1[feature][~df1[feature].isna()]
        df2_tmp = df2[feature][~df2[feature].isna()]
        if df1[feature].dtype == 'O':
            enc = LabelEncoder()
            enc.fit(df1_tmp.astype(str))
            df1_tmp = enc.transform(df1_tmp)
            df2_tmp = enc.transform(df2_tmp)
        if n_features == 1:
            sns.kdeplot(df1_tmp, label=label1, ax=ax)
            sns.kdeplot(df2_tmp, label=label2, ax=ax)
            ax.set_xlabel(feature, fontsize=9)            
            #ax.set_title(titles, fontsize=10)            
        else:
            sns.kdeplot(df1_tmp, label=label1, ax=ax[i])
            sns.kdeplot(df2_tmp, label=label2, ax=ax[i])
            ax[i].set_xlabel(feature, fontsize=9) 
            #ax[i].set_title(titles, fontsize=10)
    plt.tight_layout()     
    plt.show()
    del df1, df2, df1_tmp, df2_tmp
    gc.collect()
                       
def plot_feature_distribution_w_target(data, neg, pos, features, label1, label2, bins=10, color1='r', color2='g'):
    assert isinstance(features, list), 'Features should be list type'
    sns.set_style('darkgrid')
    n_features = len(features) 
    if n_features > 4:
        rows = int(np.ceil(n_features / 4))
        cols = 4
    else:
        rows = 1
        cols = n_features
    if n_features >= 4*7:
        plt.figure(figsize=(24, 22))
    else:
        plt.figure(figsize=(16, 14))
    i = 0    
    for f in features:
        i += 1
        neg_tmp = data[neg][f][~data[neg][f].isna()]
        pos_tmp = data[pos][f][~data[pos][f].isna()]
        if data[f].dtype == 'O':
            enc = LabelEncoder()
            enc.fit(data[f].astype('str'))
            neg_tmp = enc.transform(neg_tmp)
            pos_tmp = enc.transform(pos_tmp)   
        plt.subplot(rows, cols, i)        
        sns.distplot(neg_tmp, label=label1, kde=True, rug=False, hist=True, bins=bins, color=color1) 
        sns.distplot(pos_tmp, label=label2, kde=True, rug=False, hist=True, bins=bins, color=color2) 
        plt.xlabel(f, fontsize=9)    
        plt.legend(labels=[label1, label2])
    plt.tight_layout()     
    plt.show()
    del neg_tmp, pos_tmp
    
def plot_cate_feature_distribtion(data, feature, target):
    tmp = pd.crosstab(data[feature], data[target], normalize='index') * 100
    tmp = tmp.reset_index()
    total = len(data)    
    tmp_count = data[feature].value_counts()
    categories_name = tmp_count.index.tolist()
    
    plt.figure(figsize=(14, 6))
    plt.suptitle('{} Distributions'.format(feature), fontsize=20)
    plt.subplot(121)
    g = sns.countplot(x=feature, data=data, order=categories_name)
    g.set_title('{} Distributions'.format(feature), fontsize=15)
    g.set_xlabel("{} Name".format(feature), fontsize=15)
    g.set_ylabel("Count", fontsize=15)
    g.set_ylim(0, tmp_count[0] + tmp_count[-1])
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x() + p.get_width()/2, 
               height + 3,
               '{:1.2f}%'.format(height / total*100),
               ha="center", fontsize=14) 
        
    plt.subplot(122)
    g1 = sns.countplot(x=feature, hue=target, data=data, order=categories_name)
    plt.legend(title='{}'.format(target), loc='best', labels=data[target].unique().tolist())
    gt = g1.twinx()
    gt = sns.pointplot(x=feature, y=data[target].unique().tolist()[1], 
                       data=tmp, color='black', order=categories_name, legend=False)
    gt.set_ylabel("% of Negative target", fontsize=14)
    g1.set_title("{} by Target({})".format(feature, target), fontsize=15)
    g1.set_xlabel("{}".format(feature), fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    plt.show()    
    
def plot_interaction_w_target(data, feature1, feature2, target):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sns.boxenplot(x=feature1, y=feature2, hue=target, data=data, ax=ax)
    ax.set_title("{} by {} and {}".format(feature1, feature2, target), fontsize=16)
    ax.set_xlabel("{}".format(feature1), fontsize=17)
    ax.set_ylabel("{}".format(feature2), fontsize=17)
    plt.show()

def plot_feature_freq(data, features, target):
    assert isinstance(features, list), 'Features should be list type'        
    sns.set_style('darkgrid')
    n_feature = len(features)         
    for i, feature in enumerate(features):     
        if feature != target and data[feature].dtype == 'O':
            print(feature + ' is categorical. Categorical features not supported yet.')
        else:
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 1, 1)
            tmp = pd.crosstab(data[feature],
                              data[target], 
                              normalize='index').reset_index().rename({0:'pos', 1:'neg'}, axis=1)
            count_feature = 'c_' + feature
            tmp[count_feature] = tmp[feature].map(data[feature].value_counts(dropna=False))
            tmp_pivot = tmp[tmp.neg > 0].pivot(index=count_feature, columns=feature, values='neg')
            sns.heatmap(tmp_pivot, cmap='YlGnBu', center=0.0)
            ax.set_ylabel(count_feature, fontsize=12)
            ax.set_xlabel(feature, fontsize=12)
            plt.xticks(rotation=45)
            ax.set_title('{} / {} interaction'.format(feature, count_feature), fontsize=14)
            plt.tight_layout()
            plt.show()
            del tmp, tmp_pivot
            print('-----------------------------------------------------------------------------------------------------')
            print('\n')
    gc.collect()
        