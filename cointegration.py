# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import json,pdb,pickle
import numpy as np
import pandas as pd
from dbloader import dbloader
import os

def get_stock_codes():
    with open("./utils/stock_codes_list.txt","r") as f:
        res = json.loads(f.read())
    return res

def align_series(ts_1,ts_2):
    ts_2 = ts_2.loc[ts_1.index]
    ts_2 = ts_2.loc[ts_2.notnull()]
    ts_1 = ts_1.loc[ts_2.index]
    return ts_1,ts_2
        
def find_n_top_pvalue(stock_code="600000",n=10):
    import statsmodels.api as sm
    import seaborn as sns
    df = pd.read_csv("{}_corr_result.csv".format(stock_code))
    df = df[:n]
    codes_list = [str(s) for s in df.s2.values.tolist()]
    codes_list.append(stock_code)
    n = len(codes_list)
    db = dbloader()
    pairs = []
    pvalue_matrix = np.ones((n,n))
    for i in range(n):
        for j in range(i+1,n):
            print(i,j)
            s1 = db.load_day_a(codes_list[i],"20090101","20091130")["Closing Price"]
            s2 = db.load_day_a(codes_list[j],"20090101","20091130")["Closing Price"]
            s1,s2 = align_series(s1,s2)
            res = sm.tsa.stattools.coint(s1,s2)
            pvalue = res[1]
            pvalue_matrix[i,j] = pvalue
            if pvalue < 0.05:
                pairs.append((codes_list[i],codes_list[j],pvalue))
    
    sns.heatmap(1-pvalue_matrix,xticklabels=codes_list,
        cmap="RdYlGn_r",yticklabels=codes_list,mask=(pvalue_matrix==1))
    plt.savefig("./image/{}_pvalue_mat.png".format(stock_code))
    plt.show()
    print(pairs)
    return pvalue_matrix,pairs

def plot_spread(S1="600000",S2="600015"):
    db = dbloader("./dataset/training_data")
    ts1 = db.load_day_a(S1,"20090101","20091130")["Closing Price"]
    ts2 = db.load_day_a(S2,"20090101","20091130")["Closing Price"]
    ts1,ts2 = align_series(ts1,ts2)
    # plot scatters
    plt.scatter(ts1.values,ts2.values)
    plt.xlabel("ts1 value")
    plt.ylabel("ts2 value")
    plt.title("Price Scatter between {} and {}".format(S1,S2))
    plt.savefig("./image/scatter_{}_{}.png".format(S1,S2),dpi=300)
    plt.show()
    # plot diff
    diff = ts1 - ts2
    diff_mean = diff.mean()
    diff_std  = diff.std()
    mean_line = pd.Series(diff_mean,index=diff.index)
    up_line = pd.Series(diff_mean + diff_std,index=diff.index)
    down_line = pd.Series(diff_mean - diff_std,index=diff.index)
    sets = pd.concat([diff,mean_line,up_line,down_line],axis=1)
    sets.columns = ["diff","mean","up","down"]
    sets.plot(figsize=(14,7))
    plt.savefig("./image/diff_{}_{}.png".format(S1,S2),dpi=800)
    plt.show()

def OLS_fit(S1="600015",S2="600016"):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import summary_table
    db = dbloader("./dataset/training_data")
    ts1 = db.load_day_a(S1,"20090101","20091130")["Closing Price"]
    ts2 = db.load_day_a(S2,"20090101","20091130")["Closing Price"]
    ts1,ts2 = align_series(ts1,ts2)
    x = ts1.values
    Y = ts2.values
    X = sm.add_constant(x)
    res = sm.OLS(Y,X).fit()
    print(res.summary())
    _,data,_ = summary_table(res)
    plt.plot(Y,label="real")
    plt.plot(res.fittedvalues,label="fitted")
    plt.legend()
    plt.savefig("./image/OLS_{}_{}.png".format(S1,S2))
    plt.show()
    w1 = res.params[1]
    diff = ts2 - w1 * ts1
    diff_mean = diff.mean()
    diff_std  = diff.std()
    mean_line = pd.Series(diff_mean,index=diff.index)
    up_line = pd.Series(diff_mean + diff_std,index=diff.index)
    down_line = pd.Series(diff_mean - diff_std,index=diff.index)
    sets = pd.concat([diff,mean_line,up_line,down_line],axis=1)
    sets.columns = ["diff","mean","up","down"]
    sets.plot(figsize=(14,7))
    plt.savefig("./image/OLS_diff_{}_{}.png".format(S1,S2),dpi=800)
    plt.show()

def dig_significant_pair(n=5):
    "select n largest corr stocks" 
    import statsmodels.api as sm 
    db = dbloader("./dataset/training_data")
    code_list = get_stock_codes()
    top_corr_list = os.listdir("./top_corr")
    pairs = []
    for i,code in enumerate(top_corr_list):
        print("*" * 20)
        print("{}/{}: {}".format(i,len(top_corr_list),code))
        df_corr = pd.read_csv("./top_corr"+"/"+code)
        df_corr = df_corr[:n]
        codes_list = [str(s) for s in df_corr.s2.values.tolist()]
        codes_list.append(df_corr.s1.unique()[0])
        n = len(codes_list)
        pvalue_matrix = np.ones((n,n))
        for i in range(n):
            for j in range(i+1,n):

                s1 = db.load_day_a(codes_list[i],"20090101","20091130")["Closing Price"]
                s2 = db.load_day_a(codes_list[j],"20090101","20091130")["Closing Price"]
                s1,s2 = align_series(s1,s2)
                res = sm.tsa.stattools.coint(s1,s2)
                pvalue = res[1]
                pvalue_matrix[i,j] = pvalue
                print(i,j,"P-VALUE:",pvalue)
                if pvalue < 0.05:
                    pairs.append((codes_list[i],codes_list[j],pvalue))
        n -= 1
        with open("pairs.txt","wb") as f:
            f.write(pickle.dumps(pairs))
        print("*" * 20)
    
    s1_list,s2_list,p_list = [],[],[]
    for s1,s2,pv in pairs:
        s1_list.append(s1)
        s2_list.append(s2)
        p_list.append(pv)
    
    df_p = pd.DataFrame({"S1":s1_list,"S2":s2_list,"pvalue":p_list})
    df_p.to_csv("./rule/pairs.csv",index=False,encoding="utf-8")

def build_rule():
    "build trading rules with p-value"
    import statsmodels.api as sm 
    """
    with open("pairs.txt","rb") as f:
        pairs = pickle.loads(f.read())
    """
    db = dbloader()
    df_pair = pd.read_csv("./rule/pairs.csv") # cols=[S1,S2,pvalue]
    up_list,down_list,w_list,s1_s2 = [],[],[],[]
    for idx in df_pair.index:
        code1,code2 = str(int(df_pair.loc[idx].S1)),str(int(df_pair.loc[idx].S2))
        s1_s2.append(str(set([int(code1),int(code2)])))
        print("{}/{} {}/{}".format(idx+1,df_pair.shape[0],code1,code2))
        s1 = db.load_day_a(code1,"20090101","20091130")["Closing Price"]
        s2 = db.load_day_a(code2,"20090101","20091130")["Closing Price"]
        ts1,ts2 = align_series(s1,s2)
        x = ts1.values
        Y = ts2.values
        X = sm.add_constant(x)
        res = sm.OLS(Y,X).fit()
        w1 = res.params[1]
        diff = ts2 - w1 * ts1
        diff_mean = diff.mean()
        diff_std  = diff.std()
        up_line = diff_mean + diff_std
        down_line = diff_mean - diff_std
        up_list.append(up_line)
        down_list.append(down_line)
        w_list.append(w1)
    

    df_rule = pd.DataFrame({"up":up_list,"down":down_list,"w":w_list,"name":s1_s2},index=df_pair.index)
    df_rule = pd.concat([df_pair,df_rule],axis=1).reset_index(drop=True)
    df_rule.drop_duplicates(subset="name",inplace=True)
    df_rule = df_rule.sort_values(by="pvalue").reset_index(drop=True)
    df_rule.to_csv("./rule/trade_rule.csv",index=False,encoding="utf-8")

def main():
    code_list = get_stock_codes()
    # plot_spread("600000","600036")
    # find_n_top_pvalue("600000",5)
    # OLS_fit("600015","600016")
    dig_significant_pair(n=4)
    build_rule()

if __name__ == "__main__":
    main()



