# IEEE CIS FRAUD DETECTION

## Introduce

In this competition you are predicting the probability that an online transaction is fraudulent, as denoted by the binary target isFraud.

The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.

The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features.

## Data

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

**Card columns**
1. card 4 是卡片類別
2. card 3 有可能是 country
3. card 1/2/3/5 在 isFraud 的分佈下，有明顯的差異
4. card 3 值越高 Fraud 的機率越高
5. card 6 顯示 credit 的交易有問題機率比 debit 來的要高，資料內 debit 的母體佔約 74.5%
6. card 4 中的 discover 卡 Fraud 的機率最高，visa 與 mastercard 差不多，american express 機率最低
7. card 3/5 與 count 的交互影響有顯著

**D columns**
1. D6/7/8/9/11/12/13/14 missing% 超過 85%，但在有值得狀況下，對於 isFraud 有明顯的趨勢
2. D9 的 Fraud 分佈有明顯差異 
3. D1/2/3/4/5/7/15 都有值越高 Fraud 的比例越低的情況，有線性關係的存在
4. D3/6/7/9/12/13/14 與 count 的交互影響有顯著

**C columns**
1. 欄位定義為 counting，統計資料顯示多數欄位的值都落在 $[0, 3]$ 的區間內，而最大值都遠大於一般正常
2. C4/7/8/10/11/12 都有值越高， Fraud 的機率越高
3. train 與 test 在Quartile-75%以上的值有很大的差異
4. 除了 C3 外，其他都對 count 有明顯的影響

**M columns**
1. M4 的類別中，M2 有很高的 Fraud 佔比，而 M1 有較低的比例不是 Fraud
2. 乍看下，M 系列的變數對於 Fraud 的影響不大

**V columns**
1. 根據各個 V cols 的 Missing% 來排列，可以發現某些 V cols 的計算應該是同一個 group 不同 features 的統計值
2. 總共有 15 個 groups
3.

**Other columns (addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain)**


## Experiment
1. 在 transcation data 中實驗刪除 missing value rate 較高的 features 與不刪除的比較，在這個資料明顯看出 na 的值是有顯著影響，所以假設保有 missing value rate 較高的 features 對模型是有顯著幫助

## Concept
1. Fraud 的判斷應該要從 seller 的角度來思考，purchaser 並不知道這筆交易會出問題，只是單純的購買物品行為。
2. 對於 Fraud 來說，傾向一次交易騙取大量金額，如果是多次交易小量金額，很容易就會被用戶提報，銀行封鎖，缺乏利潤。
3.



