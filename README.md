# AML-Blacklist-Prediction

- [Link](https://tbrain.trendmicro.com.tw/Competitions/Details/11)

- 比賽說明：
此競賽將提供參賽者公開新聞資料連結與相對應的焦點人物名單，  
透過NLP演算法，精準找出AML相關新聞焦點人物，協助優化AML焦  
點人物名單的更新作業，更有機會獲得高額獎金！

- 成果：
由於第一天API沒有成功開啟，因此此比賽以**練習的心態**參加。  
扣除第一天**成績為第12名**。

## Model Archtecture:
以下是我們的架構，分為前處理的爬蟲系統及模型。

### Crawling System


### Model
在這次比賽中我們用兩種方式，來做訓練以及測試：
#### 1. NER Model -> Blacklist Model
Step1: 訓練一個能在文章中找出**名字**的**NER模型（BERT）**。  
Step2: 利用NER任務模型當作基底，利用黑名單文章對AML文章偵測模型做fine-tune。

#### 2. 
