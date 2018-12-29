import matplotlib.pyplot as plt
import json,pdb
import numpy as np
import pandas as pd
import os
from dbloader import dbloader

def get_stock_codes():
    with open("./utils/stock_codes_list.txt","r") as f:
        res = json.loads(f.read())
    return res

def align_series(ts_1,ts_2):
    ts_2 = ts_2.loc[ts_1.index]
    ts_2 = ts_2.loc[ts_2.notnull()]
    ts_1 = ts_1.loc[ts_2.index]
    return ts_1,ts_2
        
def find_n_top_corr(stock_code="600000",codes_list=[],top_n = 30):
    db = dbloader("./dataset/training_data")
    error_tag = False
    for i,code in enumerate(codes_list):
        try:
            ts_1 = db.load_day_a(stock_code,"20080101","20091130")["Closing Price"].diff(1).dropna()
            ts_2 = db.load_day_a(code,"20080101","20091130")["Closing Price"].diff(1).dropna()
            ts_2 = ts_2.loc[ts_1.index]
            ts_2 = ts_2.loc[ts_2.notnull()]
            ts_1 = ts_1.loc[ts_2.index]            
            res = np.corrcoef(ts_1.values,ts_2.values)[0,1]
        except:
            print("[WARNING]something went wrong, res is assigned 0.0!")
            res = 0.0
            error_tag = True
        if code == stock_code:
            res = 0.0
        if ts_2.shape[0] < 100: # too small series, drop it
            print("[WARNING]series are too short, skip it")
            res = 0.0

        if i == 0:
            df_pair = pd.DataFrame({"s1":[stock_code],"s2":[code],"corr":[res,]})
        else:
            df_pair = df_pair.append({"s1":stock_code,"s2":code,"corr":res},ignore_index=True)
        print("[{}]{}/{} CORR:{}".format(i,stock_code,code,res))
        
        if i > 10000:
            break

    df_pair = df_pair.sort_values(by="corr",ascending=False).reset_index(drop=True)
    df_top_n = df_pair[:top_n]
    print(df_pair)
    print(df_top_n)
    df_pair.to_csv("./top_corr/{}_corr_result.csv".format(stock_code),index=False,encoding="utf-8")
    return df_top_n

def main():
    # existed result
    exist_codes = [r[:6] for r in os.listdir("./top_corr")]
    # get codes
    code_list = get_stock_codes()

    for code in code_list:
        print("now",code)
        if code in exist_codes:
            print("find existed code:{},skip it.".format(code))
            continue
        else:
            find_n_top_corr(stock_code=code,codes_list=code_list,top_n = 30)
             


if __name__ == "__main__":
    main()

