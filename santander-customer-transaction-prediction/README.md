# Santander Customer Transaction Prediction

Kaggle Competitions : https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion

#### Conclusion：
* Features engineering 能力的考驗，本次比賽最 powerful 的 feature 是 counting
* 資料都經過特別處理，所以資料的分佈都是 normal distribution
* Fake data 的情況
    - https://www.kaggle.com/yag320/list-of-fake-samples-and-public-private-lb-split
    - 根據這位 kaggler 的分析，訓練集與測試集之間的差異在於 unique value 的分佈不同，所以假設測試資料的產生應該是由真實樣本(real samples)以及透過真實樣本的分配(distribution)所產生的合成樣本(synthetic samples)組成。如果上述假設成立，那就可以試著找出哪些資料是 synthetic samples。
    
    
* Ensemble Tree model 的學習是 split value 學習，所以如何製造出可以有高度貢獻的 split value feature 是重點
* Neural Network 學習的是線性與非線性的特點，所以在此次比賽中預測結果並沒有比 Ensemble Tree model 來得好

#### Models
1. LightGBM
    - LGBM "divides" the histogram with vertical lines because LGBM does not see horizontal differences. A histogram places multiple values into a single bin and produces a smooth picture.
    
#### Feature engineering    
1. count(var)
2. replace count(var)==1 with nan
3. var + count(var) , var - count(var), var/count(var)
4. MinMaxScaler(var)
5. MinMaxScaler(var counts)
6. reverse features: 計算 target 與 features 之間的相關性，相關係數小於 0 的，該 feature 數值 * -1
    
Reference:
* kernel:
    1. https://www.kaggle.com/cdeotte/200-magical-models-santander-0-920
    2. https://www.kaggle.com/jesucristo/magic-compilation-part-i
    3. https://www.kaggle.com/kelexu/fork-of-pytorch-nn-cyclelr-k-fold-0-920-with-aug
    4. https://www.kaggle.com/nagiss/9-solution-nagiss-part-2-2-weight-sharing-nn
    5. https://www.kaggle.com/fl2ooo/nn-wo-pseudo-1-fold-seed
    6. https://www.kaggle.com/felipemello/step-by-step-guide-to-the-magic-lb-0-922
    7. https://www.kaggle.com/titericz/giba-single-model-public-0-9245-private-0-9234

* Discussion
    1. https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003#latest-521279
    2. https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88939#latest-525018
