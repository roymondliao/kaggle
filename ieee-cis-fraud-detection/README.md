# IEEE CIS FRAUD DETECTION

## Introduce

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features.

## Data

**Note**
1. Data are collected from the world.
2. isFraud is define by the below reason:
    * chargebank on card
    * user account
    * associated email address
    * transcation directly
3. D1 is "days since client (credit card) began"   
    

**Categorical Features - Transaction**

* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
* TransactionAMT: transaction payment amount in USD
* ProductCD: product code, the product for each transaction
* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
* addr: address
* dist: distance
* P_ and (R__) emaildomain: purchaser and recipient email domain
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual * meaning is * masked.
* D1-D15: timedelta, such as days between previous transaction, etc.
* M1-M9: match, such as names on card and address, etc.
* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

**Categorical Features - Identity**
Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. They're collected by Vesta’s fraud protection system and digital security partners. (The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

Categorical Features:

* DeviceType
* DeviceInfo
* id12 - id38

## EDA 

**Base columns (TransactionDT, TransactionAmt, ProductCD)**
1. Imbalanced data, 1:96.5% / 0:3.5% 
2. train 與 test 的資料分布於不同的時間點，所以在做 cross validation 的時候需考慮 time split.
3. ProductCD 為 C 的類別， Fraud 的機率高達 12%。類別 W 與 C 所佔的數量最多，可以當作進階數據 Groupby 的分類依據
4. W/C 物品的交易都是有 cents 的 (美元的分)
5. TransactionAmt 在 train 有超過 30000 美元的數值，但在 test 最大為 10270 美元，此筆可以當作 outlier 來看，刪不刪都不影響模型

```
ProductCD -> group encoding, target encoding
TransactionAmt --> rounding
```

**Card columns**
card 都是類別資料，但用數字來表示
1. card 3 有可能指的卡片所屬的 country，透過 email domain 的資訊比對，card3 應該不是 country
2. card 1/2/3/5 在 isFraud 的分佈下，有明顯的差異
3. card 3 值越高 Fraud 的機率越高
4. card 6 顯示 credit 的交易有問題機率比 debit 來的要高，資料內 debit 的母體佔約 74.5%
5. card 4 中的 discover 卡 Fraud 的機率最高，visa 與 mastercard 差不多，american express 機率最低
6. card 4/6 適合與其他 feature 做 groupby 的整合
7. card 3/5 與 count 的交互影響有顯著

```
card1 -> target encoding
card2 -> count encoding
card3/5 -> count encoding, binning
card4/6 -> aggregated group encoding
```

**D columns**
1. D6/7/8/9/11/12/13/14 missing% 超過 85%，但在有值得狀況下，對於 isFraud 有明顯的趨勢
2. D2/9 的 Fraud 分佈有明顯差異 
3. D1/2/3/4/5/7/8/11/15 都有值越高 Fraud 的比例越低的情況，有線性關係的存在
4. D3/6/7/9/12/13/14 與 count 的交互影響有顯著

```
D1/2/3/4/5/7/8/11/15 -> binning
D9 -> group encoding: unique value 24 個，可當作 categroy 來使用
D8 -> target encoding
D3/6/7/9/12/13/14-> count encoding
以下的組別在與 count 的計算結果，圖形有相似的走勢
- D1/2/12 
- D3/5/7/11/13
- D4/6/10/15
D 系列的可以考慮用 linear model 來做 meta-feature
```

**C columns**
The actual * meaning is * masked --> 表示可能 count 數值越多的就應該是 missing value
1. 欄位定義為 counting，統計資料顯示多數欄位的值都落在 $[0, 3]$ 的區間內，而最大值都遠大於一般正常
2. C1/4/7/8/10/11/12 都有值越高， Fraud 的機率越高
3. train 與 test 在Quartile-75%以上的值有很大的差異
4. 除了 C3 外，其他都對 count 有明顯的影響

```
C1/4/7/8/10/11/12 -> binning
C13 -> binning 
C 系列除了C3 -> count encoding
```

**M columns**
1. M4 的類別中，M2 有很高的 Fraud 佔比，而 M1 有較低的比例不是 Fraud
2. M4 在 Fraud 的分佈上有明顯差異
3. M1/2/3, M/7/8/9 各為同一個問題 group
4. M5/6 有部份相同
5. 乍看下，M 系列的變數對於 Fraud 的影響不大

```
M1/2/3/5/6/7/8/9 -> polynomial encoding
M4 -> group encoding, label encoding
```

**V columns**
1. 根據各個 V cols 的 Missing% 來排列，可以發現某些 V cols 的計算應該是同一個 group 不同 features 的統計值
2. 總共有 15 個 groups

```
V columns --> 15 個 groups 使用 PCA or KNN features
```

**Other columns (addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain)**
1. addr1 + addr2 為主要地址，addr2 在數值較高的部分，Fraud 的比率高
2. dist2 missing value 的比例高達 93%
3. addr1/addr2 與 count 的交互影響有顯著
4. addr hash 後明顯地特定區間有較高的 Fraud 機率
5. email domain，在購買方面 microsoft 的 Fraud 比例較高，出售方 apple/google 的 Fraud 比例較高

**ID columns**
1. id columns 的 missing value 都大於 75% 以上，id18/23/22/27/26/21/07/08/25/24 都大於 90%
2. id13/21/26的資料分布 train 與 test 略有不同
3. id19/20/21/26/31/33 在 positive 與 negative 的分佈有差異
4. id01/02/03/04/09/10/14/22 有值越大，Fraud 機率越高的線性相關
5. id30 為手機 OS version，id3為 browser version，id33為手機解析度
6. id31 有嚴重的 covariate shift

## Experiment
1. 在 transcation data 中實驗刪除 missing value rate 較高的 features 與不刪除的比較，在這個資料明顯看出 na 的值是有顯著影響，所以假設保有 missing value rate 較高的 features 對模型是有顯著幫助


## Concept
1. Fraud 的判斷應該要從 seller 的角度來思考，purchaser 並不知道這筆交易會出問題，只是單純的購買物品行為。
2. 對於 Fraud 來說，傾向一次交易騙取大量金額，如果是多次交易小量金額，很容易就會被用戶提報，銀行封鎖，缺乏利潤。
3. 需要多考慮資料的陷阱，某些 TransactionID 可能是同一個交易者

## Soultion in competition

#### Exploratory Data Analysis
1. 觀察各 features frequency 在 train/test 的分布狀況，有些 features 透過 log transsform/target encoding 的方式可以讓分佈相似，以利模型學習到有用的資訊
2. User identification
    * Column used for identification: D1/D3/addr1/P_emaildomain/ProductID/C13
    * The identification work was performed with attention to the following two points.
        1. emaildomain: Replace anonymous.com and mail.com with NaN
        2. There is a gap period between train and test-public Recognized as the same user even if D3 etc. don't fit due to the gap period
        
#### Feature Engineering 


#### Feature Selection




