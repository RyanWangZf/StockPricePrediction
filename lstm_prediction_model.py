# -*- coding: utf-8 -*-
"""
Author: @Zifeng
required packages:
1. keras
2. pandas
3. numpy
4. sklearn
"""


import pdb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# customized data loader
from dbloader import dbloader

from keras.layers import LSTM,Dense,Layer,Input, Activation
from keras import backend as K
from keras.layers import BatchNormalization,LeakyReLU,Dropout
from keras.models import Sequential,Model

from sklearn.preprocessing import MinMaxScaler



class data_pipeline:
    """
    used for data preprocessing & feature engineering
    let diff(Closing Price,1) be the target
    """
    def __init__(self,df,timestep,scaler):
        self.timestep = timestep
        self.scaler = scaler

        # padding price_diff in 9:30 and 13:00:01 with zero
        ts = df["Closing Price"].diff(1).fillna(0.0)
        id_0930 = [pd.to_datetime(d.strftime("%Y%m%d")+"0930") 
            for d in ts.index.normalize().unique()]
        id_1301 = [pd.to_datetime(d.strftime("%Y%m%d")+"1301")
            for d in ts.index.normalize().unique()] 
        ts.loc[id_0930[1:]] = 0.0
        ts.loc[id_1301] = 0.0
        df["price_1_diff"] = ts
        self.df = df

    def create_dataset(self,fit_scaler=False):
        """
        [ATTENTION]
        datasetx: np.array
        datasety: pd.Series
        """ 
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
        return datasetx,datasety
    
    # ---
    # features
    # ---
    
    def get_std_window(self,df,n=10,feature_list=[]):
        ft_name = "price_{}_std".format(n)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].rolling(n).std()
        return df
    
    def get_avg_window(self,df,n=10,feature_list=[]):
        ft_name = "price_{}_avg".format(n)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].rolling(n).mean()
        return df
    
    def get_range_window(self,df,n=10,feature_list=[]):
        ft_name = "price_{}_range".format(n)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].rolling(n).max() - df["Closing Price"].rolling(n).min()
        return df
    
    def get_shift_target_diff(self,df,shift=1,feature_list=[]):
        ft_name = "price_diff_{}_shift".format(shift)
        feature_list.append(ft_name)
        df[ft_name] = df["Closing Price"].shift(1).diff(1)
        return df
    
    def feature_engineering(self):
        df = self.df
        feature_list = []
        
        # add new features
        df = self.get_std_window(df,n=10,feature_list=feature_list)
        df = self.get_avg_window(df,n=10,feature_list=feature_list)
        df = self.get_range_window(df,n=10,feature_list=feature_list)
        df = self.get_shift_target_diff(df,shift=1,feature_list=feature_list)
        # drop rows with NaN
        df = df.dropna()
        self.df = df 
        self.feature_list = feature_list
        return

class CustomLoss(Layer):

    def __init__(self,**kwargs):
        self.is_placeholder = True
        super(CustomLoss,self).__init__(**kwargs)

    def call(self,inputs):
        y_pred = inputs[0]
        y_true = inputs[1]
        loss = K.mean(K.square(y_true-y_pred))
        self.add_loss(loss,inputs=inputs)
        return y_pred


class LSTM_model(object):
    def __init__(self,timestep=3):
        return

    def inference(self):
        data_dim = self.data_dim
        timestep = self.timestep

        BN = BatchNormalization
        
        x = Input(shape=(timestep,data_dim),name="main_input")

        h = LSTM(32,return_sequences=True)(x)
        h = Dropout(0.2)(h)
        h = LSTM(64,return_sequences=False)(h)

        h = Dense(64)(h)
        h = BN()(h)
        h = Dropout(0.5)(h)

        h = Dense(32)(h)
        h = BN()(h)
        h = Dropout(0.5)(h)

        h = Dense(1,use_bias=False)(h)
        h = Activation("linear")(h)

        # h = CustomLoss()([h,y_true])
        model = Model(inputs=x,outputs=h)
        model.compile(loss="mse",metrics=["mse"],optimizer="rmsprop")
        
        return model
    
    def fit(self,x,y,epochs=10,batch_size=8,x_valid=None,y_valid=None):
        self.data_dim = x.shape[2]
        self.timestep = x.shape[1]
        model = self.inference()

        try:
            res = model.fit(x,y,epochs=epochs,validation_data=(x_valid,y_valid),
                batch_size=batch_size,verbose=1,shuffle=False)
        except:
            res = model.fit(x,y,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=False)
        
        # show training process
        """
        val_loss = res.history["val_loss"]
        train_loss = res.history["loss"]
        plt.plot(val_loss,label="val loss")
        plt.plot(train_loss,label="train loss")
        plt.legend()
        plt.show()
        """

        # link op
        self.model = model
        
    
    def eval(self,x_test,y_test,batch_size=32):
        "use trained model "
        return self.model.evaluate(x_test,y_test,batch_size=batch_size)
    
    def predict(self,x):
        pred = self.model.predict(x)

        return self.model.predict(x,batch_size=batch_size)

    def eval_and_plot(self,x,y,df_test,batch_size=1,scaler=None):
        """
        ATTENTION
        x: np.array
        y: pd.Series
        """
        print("*" * 20)
        print("evaluate and plot predictions...")
        # get dates list, predict on each data

        dates = [d.strftime("%Y%m%d") for d in y.index.normalize().unique()]
        # predict separately
        indx = range(y.shape[0])
        for i,date in enumerate(dates):
            raw_price_1 = df_test.loc[date+"0930"]["Closing Price"].values
            raw_price_2 = df_test.loc[date+"1301"]["Closing Price"].values
            sl_1 = y.between_time("9:30","11:30").index.get_loc(date)
            sl_2 = y.between_time("13:00","15:00").index.get_loc(date)
            x_pred_1 = x[indx[sl_1]]
            x_pred_2 = x[indx[sl_2]]
            pred_1 = raw_price_1 + pd.Series(
                self.model.predict(
                    x_pred_1,batch_size=batch_size).flatten()).cumsum().values
            pred_2 = raw_price_2 + pd.Series(
                self.model.predict(
                    x_pred_2,batch_size=batch_size).flatten()).cumsum().values
            y_true_1 = raw_price_1 + y.loc[y.index[sl_1]].cumsum().values
            y_true_2 = raw_price_2 + y.loc[y.index[sl_2]].cumsum().values
            if i == 0:
                y_true = np.r_[y_true_1,y_true_2]
                y_pred = np.r_[pred_1,pred_2]
            else:
                y_true = np.r_[y_true,y_true_1,y_true_2]
                y_pred = np.r_[y_pred,pred_1,pred_2]
        y_pred = pd.Series(y_pred.flatten(),index=y.index)
        y_pred = pd.concat([ y_pred.between_time("9:30","11:30"),
            y_pred.between_time("13:00","15:00")],axis=0)
        y_pred = y_pred.sort_index()
        
        y_true = pd.Series(y_true.flatten(),index=y.index)
        y_true = pd.concat([y_true.between_time("9:30","11:30"),
            y_true.between_time("13:00","15:00")],axis=0)
        y_true = y_true.sort_index()
        
        mse = ((y_pred.values - y_true.values)**2).mean()
        print("MSE:",mse)
        print("*" * 20)
        plt.plot(y_true.values,label="real")
        plt.plot(y_pred.values,label="pred")
        plt.legend()
        plt.savefig("./evaluation/pred.png",dpi=300)
        plt.show()

        return y_pred

def main():
    "set params"
    timestep = 5

    "load train & test"
    db = dbloader("./dataset/training_data")
    _,df_train = db.load("600000","20080101","20080201")
    _,df_valid = db.load("600000","20080602","20080701")
    _,df_test  = db.load("600000","20080202","20080210")
    
    "select col from raw data"
    cols = ["Closing Price"]
    df_train = df_train[cols]
    df_valid = df_valid[cols]
    df_test = df_test[cols]
    
    "creat data pipeline"
    # training data
    # initialize the scaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train = data_pipeline(df_train,timestep=timestep,scaler=scaler)
    x_train,y_train = data_train.create_dataset(fit_scaler=True)
    scaler = data_train.scaler

    # valid
    data_valid = data_pipeline(df_valid,timestep=timestep,scaler=scaler)
    x_val,y_val = data_valid.create_dataset(fit_scaler=False)
    
    # test
    data_test = data_pipeline(df_test,timestep=timestep,scaler=scaler)
    x_test,y_test = data_test.create_dataset(fit_scaler=False)
    
    "build LSTM model"
    model = LSTM_model(timestep=timestep)
    model.fit(x_train,y_train,epochs=50,batch_size=512,x_valid=x_val,y_valid=y_val)
    y_pred = model.eval_and_plot(x_test,y_test,df_test,batch_size=32,scaler=scaler)

if __name__ == "__main__":
    main()

    

