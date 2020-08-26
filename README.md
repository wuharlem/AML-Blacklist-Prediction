# AML-Blacklist-Prediction

- [連結](https://tbrain.trendmicro.com.tw/Competitions/Details/11)

- 比賽說明：
此競賽將提供參賽者公開新聞資料連結與相對應的焦點人物名單，  透過NLP演算法，精準找出AML相關新聞焦點人物，協助優化AML焦點人物名單的更新作業。

## Solution Archtecture:
以下是我們的架構，分為前處理的爬蟲系統及模型。

### Part 1: Crawling System
利用``Crawler.py``中的``Crawler``對主辦單位所給的網址做爬蟲，並將爬蟲結果儲存在``csv``檔案之中。  

### Part 2: Model
在這次比賽中我們用兩種方式來做訓練以及測試。在我們的測試資料集當中，平均測試分數落在**0.975**（比賽單位的評分機制）

#### 方法1：利用NER模型做fine-tune
Step1: 訓練一個能在文章中找出**名字**的**NER模型（BertForTokenClassification）**。  
Step2: 利用NER任務模型當作基底，利用黑名單文章對AML文章偵測模型做fine-tune。

step1-1: ``python model_1/NER_preprocess.py``  
step1-2: ``python model_1/NER_train.py``
```
usage: NER_train.py [-h] [--save_paths SAVE_PATHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH]
                    [--learning_rate LEARNING_RATE]

NER Model Training

optional arguments:
  -h, --help            show this help message and exit
  --save_paths SAVE_PATHS
                        the path of saved parameter
  --dataset DATASET     the path of dataset
  --batch_size BATCH_SIZE
                        batch size
  --max_epoch MAX_EPOCH
                        max number of epoch
  --learning_rate LEARNING_RATE
                        learning rate
```

step2-1: ``python model_1/blacklist_preprocess.py``  
step2-2: ``python model_1/blacklist_train.py``  

```
usage: blacklist_train.py [-h] [--save_paths SAVE_PATHS] [--dataset DATASET] [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH]
                          [--learning_rate LEARNING_RATE] [--weights WEIGHTS]

Blacklist Prediction Model 1 Training

optional arguments:
  -h, --help            show this help message and exit
  --save_paths SAVE_PATHS
                        the path of saved parameter
  --dataset DATASET     the path of dataset
  --batch_size BATCH_SIZE
                        batch size
  --max_epoch MAX_EPOCH
                        max number of epoch
  --learning_rate LEARNING_RATE
                        learning rate
  --weights WEIGHTS     weights of the model

```

##### 缺點
- 模型在判斷文章是否為AML文章並找名字的同時，除了**洗錢相關詞彙**的資訊，因為缺乏黑名單資料的情況下，會學到**名字**的資訊。
- 不同的文章分段，需要人工判斷是否有包含AML文章的資訊

#### 方法2：利用NER模型找出名字，並對其名字前後文做黑名單文章的預測
Step1: 訓練一個能在文章中找出**名字**的**NER模型（BertForTokenClassification）**。  
Step2: 將名字的前後文作為輸入文章，並將**名字統一換成李○賢**。  
Step3: 將文章丟入**Bert模型**之中做訓練，利用``[CLS]``token判斷文章是否為AML文章的分類問題。

##### 改善
- 此模型不會學習到名字的訊息，避免特定名字才會預測出黑名單的狀況

##### 缺點
- 訓練時間較長
- 不同的文章分段，需要人工判斷是否有包含AML文章的資訊
