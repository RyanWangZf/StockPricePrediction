import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
# customized data loader
from dbloader import dbloader
from keras.layers import LSTM,Dense,Layer,Input, Activation
from keras import backend as K
from keras.layers import BatchNormalization,LeakyReLU,Dropout
from keras.models import Sequential,Model,load_model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler

class data_pipeline:
    def __init__(self,df_ori,timestep):
        self.timestep = timestep
        df = df_ori.copy()
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

    def fe(self):
        df = self.df
        target = "price_1_diff"
        # dy = pd.Series(df[target].values.tolist(),name=target,index=df.index)
        dy = df[target].copy()
        dx = df.shift(241).dropna()
        dx = dx.drop([target],axis=1)
        self.dy = dy
        self.dx = dx

    def create_dataset(self,smooth_y=True):
        self.fe()
        dx = self.dx
        dy = self.dy
        timestep = self.timestep
        dxi = dx.shift(1)
        for i in range(0,timestep-1):
            # print(i)
            dx = pd.concat([dx,dxi],axis=1)
            dxi = dxi.shift(1)
        dx = dx.dropna()
        # dx = dx.loc[dy.index]
        # dy = dy.loc[dx.index]
        # normalize in window
        for idx in dx.index:
            dx.loc[idx] = (dx.loc[idx] - dx.loc[idx].mean())

        if smooth_y:
            stdy = dy.std()
            meany = dy.mean()
            dy.loc[dy.abs() > 3*stdy] = meany
        dx = dx.sort_index()
        dy = dy.sort_index()
        return dx,dy
       
def lstm(x_train,y_train,timestep=3,data_dim=1,epoch=100,batch_size=32,model_path=""):
    from keras.models import load_model 
    if len(model_path) > 2:
        print("[INFO] load existed model in",model_path)
        model = load_model(model_path)
        return model
    
    BN = BatchNormalization
    

    x = Input(shape=(timestep,data_dim),name="main_input")

    h = LSTM(32,return_sequences=True)(x)
    h = Dropout(0.2)(h)
    h = LSTM(64,return_sequences=False)(h)
    
    h = Dense(32)(h)
    h = BN()(h)
    h = LeakyReLU()(h)
    h = Dropout(0.5)(h)

    h = Dense(16)(h)
    h = BN()(h)
    h = LeakyReLU()(h) 
    h = Dropout(0.5)(h)

    h = Dense(1,use_bias=True)(h)
    h = Activation("linear")(h)
    
    model = Model(inputs=x,outputs=h)
    
    model.compile(loss="mse",metrics=["mse"],optimizer="rmsprop")
    
    res = model.fit(x_train,y_train,epochs=epoch,
                batch_size=batch_size,verbose=1,validation_split=0.05,shuffle=True)
     
    return model

def predict_on_test(lstm_m,x_test,y_test,x,df_train):
    "x is all the past training datasets"
    "x_test is only the last one day features"
    "y is used to fill the prediction, contains the date index"
    timestep = x_test.shape[1]
    dates_train = [d.strftime("%Y%m%d") for d in x.index.normalize().unique()]
    dates_train = dates_train[-30:]
    dates_test = [d.strftime("%Y%m%d") for d in y_test.index.normalize().unique()]
    x_new = df_train.copy()
    wrapper = lambda x: x.values.reshape(x.shape[0],timestep,1)
    init_price = df_train["Closing Price"][-1]
    pred_diff = y_test.copy()
    for i,date in enumerate(dates_test):
        print("predict on {}/{}".format(i,date))
        pred = lstm_m.predict(x_test)
        pred_diff.loc[date] = pred.flatten()
        pred = pd.Series(pred.flatten()).cumsum() + init_price
        init_price = pred.iloc[-1]
        y_test.loc[date] = pred.values
        x_new = pd.concat([x_new,pd.DataFrame({"Closing Price":pred.values},
            index=y_test.loc[date].index)],axis=0)
        x_new = x_new.loc[dates_train[i]:]
        data_new = data_pipeline(x_new,timestep=timestep)
        x_test,y = data_new.create_dataset()
        x_test = wrapper(x_test.loc[date])
           
    return y_test,pred_diff

def recon(pred_diff,df_test):
    dates = [d.strftime("%Y%m%d") for d in df_test.index.normalize().unique()]
    ypred =  pred_diff.copy()
    for date in dates:
        ypred.loc[date] = pred_diff.loc[date].cumsum() + df_test["Closing Price"].loc[date].iloc[0]
    return ypred

def align_x_y(x,y):
    y = y.dropna()
    x = x.loc[y.index].dropna()
    y = y.loc[x.index]
    return x,y

def main():
    train_date = ["20090601","20090831"]
    # valid_date = ["20090901","20091001"]
    test_date  = ["20090901","20091001"]
    stock_id = "600000"
    "set params"
    timestep = 20
    model_path = "./ckpt/minute_model.h5"

    "load train & test"
    db = dbloader("./dataset/training_data")
    _,df_train = db.load(stock_id,train_date[0],train_date[1])
    _,df_test  = db.load(stock_id,test_date[0],test_date[1])
    
    "select col from raw data"
    cols = ["Closing Price"]
    df_train = df_train[cols]
    df_test = df_test[cols]

    "creat data pipeline"
    wrapper = lambda x: x.values.reshape(x.shape[0],timestep,1)
    data_train = data_pipeline(df_train,timestep=timestep)
    x,y_train = data_train.create_dataset(smooth_y=True)
    x_train,y_train = align_x_y(x,y_train) # align for training
    x_train = wrapper(x_train)
    lstm_m = lstm(x_train,y_train,timestep=timestep,data_dim=1,epoch=30,batch_size=128,
        model_path=model_path)
    lstm_m.save("./ckpt/minute_model.h5")
    # test on test sets
    dates_train = [d.strftime("%Y%m%d") for d in df_train.index.normalize().unique()]
    last_date = dates_train[-1]
    x_test = wrapper(x.loc[last_date])
    y_test = pd.Series([np.nan]*df_test.shape[0],df_test.index)
    pred,pred_diff = predict_on_test(lstm_m,x_test,y_test,x,df_train)
    # reconstruct from diff to original price
    predy = recon(pred_diff,df_test)
    
    predy.name = "prediction"
    predy.plot()
    y_true = df_test["Closing Price"].copy()
    y_true.name = "groundtruth"
    y_true.plot()
    plt.legend()
    plt.savefig("./image/minute_model.png")
    plt.show()
    pdb.set_trace() 
    rmse = np.sqrt((((predy - y_true).values)**2).mean())
    print("[RMSE]:",rmse)
    
     

if __name__ == "__main__":
    main()




