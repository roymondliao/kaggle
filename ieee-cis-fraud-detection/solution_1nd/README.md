# Solution 1nd summary

**Time is not most important**
The reason adversarial validation has AUC=1 is not because the nature of fraud changes radically over time but rather because the clients in the dataset change radically over time.

**How to find uids**
Kaggle's Fraud competition dataset has 430 columns!! How do we know which columns help identify clients? In the beginning I explored columns manually and used trial and error to create UIDs. Later I developed a more systematic approach.

The training and test data have different sets of clients in each (some clients same some different) so we can find which columns help differentiate clients by performing adversarial validation. (i.e. Mix all the training and test data together. Then add a new boolean column "is_this_transaction_in_test_data?" Next train a model to classify whether a transaction is in test data or train data). If you do this on just the first 53 columns after transforming the D columns, you see AUC = 0.999 and these features as important.

To help our model generalize, we will form uid from `card1, addr1, D1n` and provide the other features as aggregations over `uid`.

The columns D4n, D10n, D15n are specific dates in time, (note D4n = day - D4) so we provide aggregated mean and standard deviation. If a specific uid has std=0 for D15n, then we know that all of its D15n are the same (unchanging). If std!=0 then that specific uid actually contains 2 or more clients and our model will split it up.

The column C13 is a cumulative count column. Therefore we prefer to aggregate nunique over uid. Then if the aggregated nunique count is the same as uid_FE where uid_FE is the number of transactions within uid, then we know this is one client. If uid_FE != AG(C13,uid,nunique) then this uid contains 2 or more clients and our model splits it up.

```python
df['D1n'] = np.floor(df.TransactionDT / (24*60*60)) - df.D1
df['uid'] = df.card1.astype(str)+'_'+df.add1.astype(str)+'_'+df.D1n.astype(str)
encode_AG(['D4n','D10n','D15n'],['uid'],['mean','std'])
encode_AG(['TransactionAmt','dist1'],['uid'],['mean','std'])
encode_AG(['C13'], ['uid'],['nunique'])
```

#### Main magic:

* client identification (uid) using card/D/C/V columns (we found almost all 600 000 unique cards and respective clients)
* uid (unique client ID) generalization by agg to remove train set overfitting for known clients and cards
    * simple uid: card1 + addr1 + addr2
* categorical features / supportive features for models
* **horizontal blend by model / vertical blend by client post-process**
    * Horizontal blend (columns): average by model, ex:(model1 + model2 + model3) / num_models
    * Vertical blend (rows): uidX (all lines/rows average of same client or card) by each or all models

#### Features validation:

We've used several validation schemes to select features:

* Train 2 month / skip 2 / predict 2
* Train 4 / skip 1 / predict 1
We were believing that it was the most stable option to predict future and unknown clients.

#### Features engineering:
* Normalize D Columns:

The D Columns are **"time deltas"** from some point in the past. We will transform the D Columns into their point in the past. This will stop the D columns from increasing with time. The formula is `D15n = Transaction_Day - D15 and Transaction_Day = TransactionDT/(24*60*60)`. And then we will take the negative of this number.

* seq2dec
    
#### Features selection:

All of these features where chosen because each increases local validation. The procedure for engineering features is as follows. First you think of an idea and create a new feature. Then you add it to your model and evaluate whether local validation AUC increases or decreases. If AUC increases keep the feature, otherwise discard the feature.

Methods:
* forward feature selection (using single or groups of features)
* recursive feature elimination (using single or groups of features)
    * https://www.kaggle.com/nroman/recursive-feature-elimination
* permutation importance
* adversarial validation
* correlation analysis
* time consistency
* client consistency
* train/test distribution analysis

One interesting trick called `time consistency` is to train a single model using a single feature (or small group of features) on the first month of train dataset and predict isFraud for the last month of train dataset. This evaluates whether a feature by itself is consistent over time. 95% were but we found 5% of columns hurt our models. They had training AUC around 0.60 and validation AUC 0.40. **In other words some features found patterns in the present that did not exist in the future.** Of course the possible of interactions complicates things but we double checked every test with other tests.

####  Models:
We had 3 main models (with single scores):

* Catboost (0.963915 public / 0.940826 private)
* LGBM (0.961748 / 0.938359)
* XGB (0.960205 / 0.932369)
Simple blend (equal weights) of these models gave us (0.966889 public / 0.944795 private). It was our fallback stable second submission.

The key here is that each model was predicting good a different group of uids in test set:

* Catboost did well on all groups
* XGB - best for known
* LGBM - best for unknown

#### Validation Strategy

We never trusted a single validation strategy so we used lots of validation strategies. Train on first 4 months of train, skip a month, predict last month. We also did train 2, skip 2, predict 2. We did train 1 skip 4 predict 1. We reviewed LB scores (which is just train 6, skip 1, predict 1 and no less valid than other holdouts). We did a CV GroupKFold using month as the group. We also analyzed models by how well they classified known versus unknown clients using our script's UIDs.

For example when training on the first 5 months and predicting the last month, we found that our

* XGB model did best predicting known UIDs with AUC = 0.99723
* LGBM model did best predicting unknown UIDs with AUC = 0.92117
* CAT model did best predicting questionable UIDs with AUC = 0.98834

Questionable UIDs are transactions that our script could not confidently link to other transactions. When we ensembled and/or stacked our models we found that the resultant model excelled in all three categories. It could predict known, unknown, and questionable UIDs forward in time with great accuracy !!

**Time consistency**
```
# ADD MONTH FEATURE
import datetime
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
train['DT_M'] = train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
train['DT_M'] = (train['DT_M'].dt.year-2017)*12 + train['DT_M'].dt.month 

# SPLIT DATA INTO FIRST MONTH AND LAST MONTH
train = train[train.DT_M==12].copy()
validate = train[train.DT_M==17].copy()

# TRAIN AND VALIDATE
lgbm = lgb.LGBMClassifier(n_estimators=500, objective='binary', 
        num_leaves=8, learning_rate=0.02)
h = lgbm.fit(train[[col]], train.isFraud, eval_metric='auc', 
        eval_set=[(train[[col]],train.isFraud),(validate[[col]],validate.isFraud)])
auc_train = np.round(h._best_score['valid_0']['auc'],4)
auc_val = np.round(h._best_score['valid_1']['auc'],4)
```

Run this with variable col equal to the column you wish to test. Here are some results. Variable C3 has auc_train = 0.5 and auc_val = 0.5. Variable C7 has auc_train = 0.65 and auc_val = 0.67. And Variable id_38 has auc_train = 0.61 and auc_val = 0.36. The next step is to remove weak variables from your model and then evaluate your entire model with your normal local validation to see if AUC increases or decreases. Using this, we found and removed 19 features and resultantly improved our model.

#### Predictions:

* 6 folds / GroupKfold by month
* almost no fine-tuning for models


#### Q/A

##### Why the M means and D means/Std by card-addr combination work?
Great questions. Let me share my opinions. Doing group aggregations with standard deviations (specifically of normalized D columns) allows your model to find clients. And doing group aggregation with means (and/or std) allows your model to classify clients. Let me explain.

Consider a group that all have the same `uid = card1_addr1_D1n` where `D1n = day - D1`. This group may contain multiple clients (credit cards). The features `D4n, D10n, and D15n` are more specific than D1n and better at finding individual clients. Therefore many times a group of `card1_addr1_D1n` will have more than 1 unique value of D15n inside suggesting multiple clients. But some groups have only 1 unique value of D15n inside suggesting a single client. When you do `df.groupby('uid').D15n.agg(['std'])` you are will get `std=0` if there is only one D15n inside and your model will be more confident that that uid is a single client (credit card).

The M columns are very predictive features. For example if you train a model on the first month of data from train using just M4 it will reach train AUC = 0.7 and it can predict the last month of train with AUC = 0.7. So when you use `df.groupby('uid').M4.agg(['mean'])` after (mapping M4 categories to integers), it allows your model to use this feature to classify clients. Now all uids with `M4_mean = 2` will be split one way in your tree and all with `M4_mean = 0` another way.

##### What is mean **train our models to find UIDs**? What do you mean by this and how is this different than writing script which finds uids using card, addrs and Ds?

An actual client group is smaller than `uid1 = card1+addr1+D1n` where `D1n=day-D1`. But as the architect we shouldn't force our model to use a more precise uid such as `uid2 = card1+addr1+D1n+D15n`. Instead we should give it uid1 and D15n separately and let the model decide when it wants to combine the two. Furthermore we can give it columns dist1, D4, D10, D15, C13, C14, V127, V136, V307, V309, V320 as various aggregations over uid1.

For example, C14 should stay the same for a single client (card holder) most of the time. In our script we enforce that is does all the time. But we give our model `df.groupby('uid').C14.agg(['std'])`. When `std=0`, the model knows that the C14 is the same for the entire uid. When `std!=0`, the model knows that that uid contains more than 1 client. Next the model will use another column to separate that uid into 2 or more clients. You see the model will figure out how to use C14 better than the rules we enforce in our script.

It wouldn't a bad thing if you gave your model uid2 = card1+addr1+D1n+D15n in addition to uid. That's what Konstantin does. And it works great. I think it would be a bad thing if we created a very precise uid and only gave it that. Then it would have no choice but to use the uid that a human created.

So I prefer giving it uid and additional "building blocks". The featureD15n = day - D15 helps because transactions that have the same D15n are more likely to belong to the same client. Thus D15n is a building block. Another building block is the feature X_train['outsider15'] = (np.abs(X_train.D1-X_train.D15)>3).astype('int8'). This boolean helps your model compare D15n to D1n. If these two numbers are within plus or minus one of each other that adds confidence that uid is correct. Additionally give your model df.groupby('uid').D15n.agg(['mean','std']). Another example is that C10 is usually increasing so you can give it df.groupby(['uid'])['C10'].shift() which should always be zero or one. If it is negative than uid contains a C10 that decreases in time.

So in conclusion, i thinks it is important to know which columns help identify clients and then transform them and/or interact them with the others in a way that LGBM can utilize to create some "black box" rule about which transactions are from the same client (credit card).

##### What is mean `client consistency` and `train/test distribution consistency`?

"train/test distribution consistency" is comparing the distribution (histogram) of a feature in the training set with its distribution (histogram) in the test set. For example TransactionDT has low values in train and high values in test. And the values do not overlap. This feature is bad (obviously) so we drop it. You can either inspect features manually or use a library like from scipy.stats import ks_2samp. Documentation here, drop features with p=0.

"client consistency" is similar to "time consistency". Split your train dataset by UIDs. Put 80% of UIDs (clients i.e. credit cards) into a train subset and 20% UIDs into a holdout (validation set). Make sure that no UID (client/credit card) is in both sets. Then train models on the 80% train and predict the 20% unseen clients. This evaluates how well features can predict unseen clients (credit cards).

## Reference
1. https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308
2. https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
3. https://www.kaggle.com/c/ieee-fraud-detection/discussion/111510


