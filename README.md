# Stockprice Prediction & Arbitrage Trading # 
## Attention ##
As the datasets are too large, we do not upload them on github, u can find a sample of datafile what we use under the `./dataset/training_data/600000.csv`.
 
## Toolkits ##
### Data Preprocessing ###
**split_train_data.py** helps extract the original dataset file traing_data.zip and split them with respect to stock codes, saving them under `./dataset/training_data`  
### Database loader ###
**dbloader.py** helps load stocks data via its stock code and assigned date interval  

## Task1: Prediction ##
### Day model ###
Runing for day price prediction: **day_model.py**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/prophet7-8.png" width=375>  
**Seasonal Components of the day model**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/component.png" width=375>  

### Minute Model ###
Runing for minute price prediction: **minute_model**.py  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/minute_model.png" width=375>  
**Model Structure**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/model.png" width=375>  

## Task2: Trading Strategy ##
### Step1 ###
Search for high correlated stock pairs, saved under `./top_corr/` runing: **find_corr_top.py**  
### Step2 ###
Search for significant stock pairs, saved under `./rule/trade_rule.csv` runing: **cointegration.py**  
**Spread price of pair(600015,600016) with high significant**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/diff_600015_600016.png" width=375>  
**OLS fit on Spread price of pair(600015,600016)**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/OLS_600015_600016.png" width=375>  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/OLS_diff_600015_600016.png" width=375>  

**A pvalue(1-pvalue) test matrix of 600000 with some other stocks**  
<img src="https://github.com/RyanWangZf/StockPricePrediction/raw/master/image/600000_pvalue_mat.png" width=375>  

## Required Packages ##
### Public ###
numpy,pandas,matplotlib  
### In Task1 ###
**day_model.py**: fbprophet==0.3.post2  
**minute_model.py**: keras, tensorflow, sklearn  

### In Task2 ###
**find_corr_top.py**: None except Public packages  
**cointegration.py**: statsmodels, seaborn  








