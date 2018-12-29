# -*- coding: utf-8 -*-

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from dbloader import dbloader
import pandas as pd
import numpy as np
import pdb,json
import matplotlib.pyplot as plt

def get_stock_codes():
    with open("./utils/stock_codes_list.txt","r") as f:
        res = json.loads(f.read())
    return res

def df_wrapper(df,log=False,diff=False):
    "make df accessible to Prophet()"
    df = df.reset_index()
    df.columns = ["ds","y"]
    if log:
        df.y = np.log(df.y.values+1e-10)
    if diff:
        df_raw = df.copy()
        df.y = df.y.diff(1).fillna(0.0).values
        return df,df_raw
    else:
        return df,None

def training_model(stock_id="600000",
    train_date=["20080101","20080201"],
    valid_date=["20080202","20080301"],
    test_date=["20080202","20080301"]):

    "load data set"
    db = dbloader("./dataset/training_data")
    df_train = db.load_day_m(stock_id,train_date[0],train_date[1])
    df_valid = db.load_day_m(stock_id,valid_date[0],valid_date[1])
    df_test  = db.load_day_m(stock_id,test_date[0],test_date[1])
    
    "select col from raw data"
    cols = ["Closing Price"]
    log_flag = False
    diff_flag = True
    df_train,raw_train = df_wrapper(df_train[cols],log=log_flag,diff=diff_flag)
    df_valid,raw_valid = df_wrapper(df_valid[cols],log=log_flag,diff=diff_flag)
    df_test,raw_test = df_wrapper(df_test[cols],log=log_flag,diff=diff_flag)

    # build model
    model = Prophet(n_changepoints = 25,
        changepoint_range=0.8,changepoint_prior_scale=0.01,daily_seasonality=False,
        weekly_seasonality=True,yearly_seasonality=False,
        seasonality_mode="additive",seasonality_prior_scale=0.005,mcmc_samples=0,
        uncertainty_samples=1000)
    # add seasonality
    model.add_seasonality(name="monthly",period=30,fourier_order=5)
    model.add_seasonality(name="dualmonthly",period=60,fourier_order=10)
    model.add_seasonality(name="quarterly",period=90,fourier_order=15)
    # add holiday
    # model.add_country_holidays(country_name="US")
    model.fit(df_train)
    future = model.make_future_dataframe(freq="D",periods=30,include_history=False)
    pred = model.predict(future)
    ypred = pd.Series(pred.yhat.values,index=pred.ds)
    ytrue = pd.Series(df_valid.y.values,index=df_valid.ds)
    if log_flag:
        ypred = np.exp(ypred.values)
        ytrue = np.exp(ytrue.values)
    elif diff_flag:
        init_date = ytrue.index[0]
        ypred.loc[init_date] = 0.0
        ypred = ypred.loc[init_date:].cumsum() + raw_valid.loc[raw_valid.ds==init_date].y.values
    ytrue = pd.Series(raw_valid.y.values,index=raw_valid.ds)
    mse = np.mean((ypred.loc[ytrue.index].values- ytrue.values)**2)
    print("*" * 20)
    print("[MSE]:",mse)
    print("*" * 20)

    plt.plot(ypred,label="pred")
    plt.plot(ytrue,label="real")
    plt.legend()
    plt.savefig("./image/prophet.png",dpi=300)
    plt.show()
    
    # pred fig
    fig = model.plot(pred)
    plt.show(fig)
    # component fig
    fig = model.plot_components(pred)
    plt.savefig("./image/component.png",dpi=300)
    plt.show(fig)
    # add changepoint
    fig = model.plot(pred)
    a = add_changepoints_to_plot(fig.gca(),model,pred)
    plt.show()

    pdb.set_trace()

def main():
    codes = get_stock_codes()
    training_model(codes[0],train_date=["20080101","20090901"],
        valid_date=["20090902","20091001"],
        test_date=["20091002","20091101"])


if __name__ == "__main__":
    main()


