import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

def plot_feature_histograms(df1, df2, label1, label2, features, bins=10, color1='coral', color2='mediumseagreen'):
    sns.set_style('darkgrid')
    i = 0
    rows = int(np.ceil(len(features) / 5))    
    plt.figure()
    fig, ax = plt.subplots(rows, 5, figsize=(18, 22))
    for feature in features:
        i += 1
        plt.subplot(rows, 5, i)        
        sns.distplot(df1[feature], label=label1, kde=False, rug=False, hist=True, bins=bins, color=color1) 
        sns.distplot(df2[feature], label=label2, kde=False, rug=False, hist=True, bins=bins, color=color2) 
        plt.xlabel(feature, fontsize=9)    
        plt.legend(labels=[label1, label2])
        locs, labels = plt.xticks()        
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-5)
        plt.tick_params(axis='y', which='major', labelsize=10)
    plt.show()

def plot_feature_distribution(df1, df2, label1, label2, features):
    sns.set_style('darkgrid')
    i = 0
    rows = int(np.ceil(len(features) / 5))    
    plt.figure()
    fig, ax = plt.subplots(rows, 5, figsize=(18, 22))
    for feature in features:
        i += 1
        plt.subplot(rows, 5, i)
        df1[feature] = df1[feature].fillna(-999)
        df2[feature] = df2[feature].fillna(-999)
        if df1[feature].sum() != 0 and df1[feature].sum() != df1.shape[0]: 
            sns.kdeplot(df1[feature], label=label1)
        if df2[feature].sum() != 0 and df2[feature].sum() != df2.shape[0]: 
            sns.kdeplot(df2[feature], label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=10)
    plt.show()
                       
def plot_feature_distribution_w_target(data, f_target, t_target, features, label1, label2, bins=10, color1='r', color2='g'):
    sns.set_style('darkgrid')
    n_features = len(features) 
    if n_features > 5:
        rows = int(np.ceil(n_features / 5))
        cols = 5
        plt.figure()
        fig, ax = plt.subplots(rows, cols, figsize=(16, 14))
    else:
        rows = 1
        cols = n_features        
        plt.figure()
        fig, ax = plt.subplots(rows, cols, figsize=(10, 8))    
    i = 0    
    for f in features:
        i += 1
        plt.subplot(rows, cols, i)        
        sns.distplot(data[f_target][f], label=label1, kde=True, rug=False, hist=True, bins=bins, color=color1) 
        sns.distplot(data[t_target][f], label=label2, kde=True, rug=False, hist=True, bins=bins, color=color2) 
        plt.xlabel(f, fontsize=9)    
        plt.legend(labels=[label1, label2])
        locs, labels = plt.xticks()        
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-5)
        plt.tick_params(axis='y', which='major', labelsize=10)
    plt.show() 
    
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
    plt.legend(title='{}'.format(target_col), loc='best', labels=data[target_col].unique().tolist())
    gt = g1.twinx()
    gt = sns.pointplot(x=feature, y=data[target_col].unique().tolist()[1], 
                       data=tmp, color='black', order=categories_name, legend=False)
    gt.set_ylabel("% of positive target", fontsize=14)
    g1.set_title("{} by Target({})".format(feature, target_col), fontsize=15)
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