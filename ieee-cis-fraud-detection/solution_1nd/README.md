# Solution 1nd summary

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
* Normalize D Columns
The D Columns are **"time deltas"** from some point in the past. We will transform the D Columns into their point in the past. This will stop the D columns from increasing with time. The formula is `D15n = Transaction_Day - D15 and Transaction_Day = TransactionDT/(24*60*60)`. And then we will take the negative of this number.
    
#### Features selection:

All of these features where chosen because each increases local validation. The procedure for engineering features is as follows. First you think of an idea and create a new feature. Then you add it to your model and evaluate whether local validation AUC increases or decreases. If AUC increases keep the feature, otherwise discard the feature.

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

#### Predictions:

* 6 folds / GroupKfold by month
* almost no fine-tuning for models
