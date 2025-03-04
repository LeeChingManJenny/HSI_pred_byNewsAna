 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime

def get_yfdata(str_code = 'HSI',startd = '2015-01-05',endd = 0,rename=True):
    if endd == 0:
        endd = datetime.today().strftime('%Y-%m-%d')
        
    data = yf.download(f'^{str_code}', start= startd, end= endd)
    
    if rename == True:
        data.columns = [f'{str_code}{i+1}' for i in range(len(data.columns))]
        
    return data

def merge_df(on, how, fillNa = 1,  *dfs):
    
    i= 0
    for df in dfs:
        if i == 0:
            output = df
        else:
            output = pd.merge(output, df, on = on, how = how)
        i = i+1
    if fillNa == 1:
        output = output.fillna(method='ffill')
    
    return output
#%%
def shift_data(orgin_df,shifts,y):
    df = pd.DataFrame()
    df['y'] = orgin_df[y]
    num = len(orgin_df.columns)
    
    for i in range(shifts):
        tmp = orgin_df.shift(i+1)
        tmp.columns = [f'x{k+1}' for k in range((i*num),((i+1)*num))]
        df = pd.concat([df,tmp],axis=1)
    
    df = df[shifts:].reset_index(drop=True)
    return df

def exp_smooth(y, a = 0.8):
    y = np.array(y)
    y1 = y
    
    for i in range(2,len(y)):
        y[i] = a*y[i]+(1-a)*y[i-1]
    
    y1[1:] = y[:-1]
    return y1
def split_train_test(df, days = 0, test_ratio = 0.2):
    training_ratio = 1 - test_ratio
    
    train_size = int(training_ratio * len(df))
    test_size = int(test_ratio * len(df))
    print(f"train_size: {train_size}")
    print(f"test_size: {test_size}")
    
    if days == 0:
        train = df[:train_size].reset_index(drop=True)
        test = df[train_size:].reset_index(drop=True) 
        return train, test
    
    else:
        tmp = len(df) - days
        train = df[:train_size].reset_index(drop=True)
        test = df[train_size:tmp].reset_index(drop=True)
        real = df[tmp:].reset_index(drop=True)
        return train, test, real

def calRmse(y_true, y_pred):
   
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def calMape(y_true, y_pred):
    
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    #print(mape)
    return mape

def oneday_predict(df,shifts, begin = 1, cus_data = 0):
    
    if cus_data == 0:
        tmp = begin*-1
        arr = np.array(df.iloc[tmp])
        for i in range(begin+1,shifts+begin):
            tmp = np.array(df.iloc[-1*i])
            arr = np.concatenate([arr,tmp])
    else:
        arr = cus_data
        for i in range(begin,shifts+begin-1):
            tmp = np.array(df.iloc[-1*i])
            arr = np.concatenate([arr,tmp])
    
    arr = pd.DataFrame(arr.reshape(1, -1))
    arr.columns = [f'x{i+1}' for i in range(len(arr.columns))]
    return arr


def displayMAPE(score, model, train_test = 0):
    if train_test == 0:
        print(f'The train MAPE of {model} is: {score}')
        
    if train_test == 1:
        print(f'The test MAPE of {model} is: {score}')
        
    return


