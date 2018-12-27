# -*- coding: utf-8 -*-

import json
import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from keras.layers import LSTM,Dense,Layer,Input, Activation
from keras import backend as K
from keras.layers import BatchNormalization,LeakyReLU,Dropout
from keras.models import Sequential,Model,load_model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler

# database loader
from dbloader import dbloader

class data_pipeline:
    def __init__(self,df,timestep,scaler):
        self.timestep = timestep
        self.scaler = scaler
        ts = df["Closing Price"].diff(1).fillna(0.0)
        df["price_1_diff"] = ts
        self.df = df
        self.index = df.index.copy()

    def get_shift_target(self,df,shift=1,feature_list=[]):
        ft_name = "price_{}_shift".format(shift)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].shift(shift)
        return df

    def get_shift_target_diff(self,df,shift=1,feature_list=[]):
        ft_name = "price_diff_{}_shift".format(shift)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].shift(shift).diff(1)
        return df

    def feature_engineering(self):
        df = self.df
        feature_list = []
        df = self.get_shift_target_diff(df,shift=1,feature_list=feature_list)
        # drop rows with NaN
        df = df.dropna()
        self.df = df 
        self.feature_list = feature_list

    def create_dataset(self,fit_scaler=False,smooth=True):
        """
        [outputs]
        datasetx: np.array
        datasety: pd.Series
        smooth: fill ultimate y value with mean
        """
        # do feature engineering at first
        self.feature_engineering()
        df = self.df
        timestep = self.timestep
        datasety = df["price_1_diff"]
        df = df[self.feature_list]# align datasety
        if fit_scaler: # fit_transform features
            df = self.scaler.fit_transform(df)
        else: # transform features
            df = self.scaler.transform(df)
        df = pd.DataFrame(df,columns=self.feature_list)
        for counter,i in enumerate(range(timestep,len(df))):
            this_ = df.iloc[i-timestep:i]
            if counter == 0:
                datasetx = this_.values.reshape(1,this_.shape[0],this_.shape[1])
            else:
                datasetx = np.r_[datasetx,this_.values.reshape(1,this_.shape[0],
                    this_.shape[1])]
        datasety = datasety[timestep:]
        # smooth
        if smooth:
            datasety[datasety.abs() > datasety.std()] = datasety.mean()
        return datasetx,datasety

def get_stock_codes():
    with open("./utils/stock_codes_list.txt","r") as f:
        res = json.loads(f.read())
    return res

def training_model(stock_id="600000",epoch=30,batch_size=16,
    train_date=["20080101","20080201"],
    valid_date=["20080202","20080301"],
    test_date=["20080202","20080301"],
    model_name="model"):
    
    "set params"
    timestep = 2

    "load data set"
    db = dbloader("./dataset/training_data")
    df_train = db.load_day(stock_id,train_date[0],train_date[1])
    df_valid = db.load_day(stock_id,valid_date[0],valid_date[1])
    df_test  = db.load_day(stock_id,test_date[0],test_date[1])
    
    "select col from raw data"
    cols = ["Closing Price"]
    df_train = df_train[cols]
    df_valid = df_valid[cols]
    df_test = df_test[cols]
   
    "create data pipeline"
    # training data
    # initialize the scaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train = data_pipeline(df_train,timestep=timestep,scaler=scaler)
    x_train,y_train = data_train.create_dataset(fit_scaler=True,smooth=True)
    scaler = data_train.scaler
        
    # valid
    data_valid = data_pipeline(df_valid,timestep=timestep,scaler=scaler)
    x_val,y_val = data_valid.create_dataset(fit_scaler=False,smooth=False)
    
    # test
    data_test = data_pipeline(df_test,timestep=timestep,scaler=scaler)
    x_test,y_test = data_test.create_dataset(fit_scaler=False,smooth=False)

    "build LSTM model"
     
    pdb.set_trace()
    
def main():
    stock_codes = get_stock_codes()
    stock_id = stock_codes[0]
    # DEFINE training plan
    training_model(stock_id,epoch=10,batch_size=256,train_date=["20080101","2009901"],
        valid_date=["20091001","20091031"],
        test_date=["20091101","20091130"],model_name="day_{}".format(stock_id))



if __name__ == "__main__":
    main()




