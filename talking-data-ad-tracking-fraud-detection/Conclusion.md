# Talkingdata-adtracking-fraud-detection

## Solution

#### Data Processing Ideas
* Up train speed
    * Extracted features from entire data and merge to sub-sampled data. Use sub-sampled data to train model that see the performances.
* Data split
    * split data by days
        * days [7, 8] to train and days [9] to validation.
        * days [7, 8, 9] to train in some model and days [8, 9] to train others.
    * split data by hours  
        * training on day <= 8
        * validating on both day 9 - hour 4, and day-9, hours 5, 9, 10, 13, 14. 
* Down-samplig
    *  This technique allowed us to use hundreds of features while keeping LGB training time less than 30 minutes. 
   
*  
    

#### Feature engineering
**Create more features as possible**
 
* Statistic featurs
    * count by several groups
    * cumcount by several groups 
    * unique count by several groups
    * mean by several groups
    * var by several groups
* Time features 
    * days
    * minute
    * second
    * time-delta

* Special features 
    * next click
    * previous click
    * time\_diff\_k\_for and time\_diff\_k\_back
        * time\_diff\_1\_for means the click time difference with next click, time\_diff\_1\_back means the click time difference with previous click. time\_diff\_2\_for means the click time difference with next next click. **when k = 1, 2, it is very effective.**
    *  top10_hoge
        * For example, if specific ip has 100 samples and the distribution of the device is 2(30 samples), 3(25 samples), 5(20 samples), 7(15 samples), 8(10 samples), nunique\_counts is 5, nunique\_counts\_ratio is 0.02, top\_counts is 30, top_counts\_ratio is 0.3, top2\_counts is 25, top2\_counts\_ratio is 0.25, top3\_counts is 15, top3\_counts\_ratio is 0.15. top\_device is 2, top2\_device is 3, top3\_device is 5, top4\_device is 7, top5\_device is 8. 
    * Matrix factorization
        * This was to capture the similarity between users and app. I use several of them. They all start with the same approach; construct a matrix with log of click counts. I used: ip x app, user x app, and os x device x app. These matrices are extremely sparse (most values are 0). For the first two I used truncated svd from sklearn, which gives me latent vectors (embeddings) for ip and user. For the last one, given there are 3 factors, I implemented libfm in Keras and used the embeddings it computes. I used between 3 and 5 latent factors. All in all, these embeddings gave me a boost over 0.0010. I think this is what led me in top 10. I got some variety of models by varying which embeddings I was using.
    
    * average attributed ratio of past clicks 
    * click count within next one/six hours 
    * categorical feature embedding by using LDA/NMF/LSA

#### Feature selection
* Forward selection
* Backward selectection
* Feature importance
    

#### Models
* Model algoritms
    * LightGBM
    * XGBoost 
    * NN
* Threshold trade off
    * IF pred >= 0.99 then 1 otherwise 0
    * IF pred >= 0.997 then 1 otherwise 0
    * IF pred >= 0.999 then 1 otherwise 0     

#### Optimize parameters
* Bayasian Optimization

#### Ensemble
* Data
    * Split n-chunck of data to train models 
* Model
    * blending
        * avg
        * hill climbing(geometric averaging)
    * stacking
        * use cross-validation to stack data 

#### System
* BigQuery on GCP
    * https://gist.github.com/tkm2261/1b3c3c37753e55ed2914577c0f96d222 
* Swap
    * 使用swap來做實體memory與虛擬memory的交換使用  

## Referece
Solution:

1. https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56406
2. https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56319
3. https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56325
4. https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56317
5. https://github.com/CuteChibiko/TalkingData
6. https://github.com/jfpuget/LibFM_in_Keras
7. [#One](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)
9. [#two](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328)
9. [#Three](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262)
10. [All](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56481)

Data Thinking:

1. https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56268
 
