"""
a database loader.
Author:@Zifeng
"""

# -*- coding: utf-8 -*-

import os
import pdb
import pandas as pd

import datetime

class dbloader:
    """
    from dbloader import dbloader
    db = dbloader("Your database path")
    _,df = db.load(stock_id,startdate,enddate)
    """
    def __init__(self,database_path="./dataset/training_data"):
        self.database_path = database_path
    
    def load(self,stock_code="600001",start_date="20080101",
        end_date="20080105"):
        file_path = self.database_path + "/" + str(stock_code) + ".csv"
        df = pd.read_csv(file_path,index_col=0,parse_dates=True)
        start,end = self.get_date_list(start_date,end_date)
        idx = pd.date_range(start=start,end=end,freq="min")
        df = df.loc[idx]
        df1 = pd.concat([df.between_time("9:30","11:30"),
            df.between_time("13:00","15:00")],axis=0)
        df_nonull = df1.dropna(how="all")
        return df1,df_nonull

    def get_date_list(self,start_date,end_date):
        start = datetime.datetime.strptime(start_date,"%Y%m%d")
        end = datetime.datetime.strptime(end_date,"%Y%m%d")
        return start,end

def main():
    db = dbloader()
    _,df = db.load("600000","20080101","20080201")
    print(df)
    pdb.set_trace()
    

if __name__ == "__main__":
    main()






