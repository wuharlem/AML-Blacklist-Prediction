# AML-Blacklist-Prediction

- [連結](https://tbrain.trendmicro.com.tw/Competitions/Details/11)

- 比賽說明：
此競賽將提供參賽者公開新聞資料連結與相對應的焦點人物名單，  
透過NLP演算法，精準找出AML相關新聞焦點人物，協助優化AML焦  
點人物名單的更新作業。

## Model Archtecture:
以下是我們的架構，分為前處理的爬蟲系統及模型。

### Crawling System
利用``Crawler.py``中的``Crawler``對主辦單位所給的網址  
做爬蟲，並將爬蟲結果儲存在``csv``檔案之中。  

### Model
在這次比賽中我們用兩種方式，來做訓練以及測試：
#### Model 1：利用NER模型做fine-tune
Step1: 訓練一個能在文章中找出**名字**的**NER模型（BertForTokenClassification）**。  
Step2: 利用NER任務模型當作基底，利用黑名單文章對AML文章偵測模型做fine-tune。

**缺點**
- 模型在判斷文章是否為AML文章並找名字的同時，除了**洗錢相關詞彙**的資訊，  
  因為缺乏黑名單資料的情況下，會學到**名字**的資訊。

- 不同的文章分段，需要人工判斷是否有包含AML文章的資訊

#### Model 2：利用NER模型找出名字，並對其名字前後文做黑名單文章的預測
Step1: 訓練一個能在文章中找出**名字**的**NER模型（BertForTokenClassification）**。  
Step2: 將名字的前後文作為輸出文章，並將**名字統一換成李○賢**。  
Step3: 將文章丟入**Bert模型**之中做訓練，利用**[CLS]**token判斷文章是否為AML文章的分類問題。

**優點**
- 此模型不會學習到名字的訊息，避免特定名字才會預測出黑名單的狀況

**缺點**
- 訓練時間較長
- 不同的文章分段，需要人工判斷是否有包含AML文章的資訊
