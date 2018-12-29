"""
Split the raw large csv file into files of each stock code.
Author: @Zifeng
"""

# -*- coding: utf-8 -*-


import pandas as pd

import os
import pickle
import pdb
import time

def pickle_write(file_path,x):
    with open(file_path,"wb") as f:
        f.write(pickle.dumps(x))
    return

def pickle_read(file_path,x=None):
    with open(file_path,"rb") as f:
        x = pickle.loads(f.read())
    return x

def combine_date_and_time(df):
    ts = pd.to_datetime(df["Date"]+" "+df["Time"])
    df = df.drop(["Date","Time"],axis=1)
    df.index = ts
    return df

def split_by_code(df,code_list,save_dir):
    save_dir_list = os.listdir(save_dir)

    for code in code_list:
        code_file_name = str(code) + ".csv"
        save_path = "./{}/{}".format(save_dir,code_file_name)
        df_code = df[df["Stock Code"] == code]

        if code_file_name in save_dir_list:
            print("Find existed {} file, adding new items.".format(code))
            df_code_old = pd.read_csv(save_path,index_col=0,parse_dates=True) 
            df_code = pd.concat([df_code_old,df_code],axis=0)
        
        # drop duplicated index
        df_code = df_code[~df_code.index.duplicated()]
        df_code.to_csv(save_path,index=True,encoding="utf-8")
        
def run_split(save_dir="training_data"):
    # reading chunk.csv, please use pd.read_csv("chunk0.csv",index_col=0,parse_dates=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reader = pd.read_csv("training_data.zip",compression="zip",chunksize=1000000)
    for i,chunk in enumerate(reader):
        print("*" * 20)
        print(time.strftime("%Y.%m.%d %H:%M:%S",time.localtime(time.time())))
        print("coping with {}-th chunk".format(i))
        chunk = combine_date_and_time(chunk)        
        code_list = chunk["Stock Code"].unique()
        print("stock code list:",code_list)
        split_by_code(chunk,code_list,save_dir) 
        print("*" * 20)

def main():
    run_split("training_data")


if __name__ == "__main__":
    main()




