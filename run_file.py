# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:21:16 2025

@author: USER
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import news_ana
import numpy as np
import torch

stock_df, news_grouped, dataset,dataloader,stock_feat = news_ana.read_data(update = 0)
train_dataset, val_dataset, train_loader,val_loader = news_ana.train_test_split(stock_df, news_grouped)

news_ana.train_finin(train_loader,val_loader,EPOCHS = 100, save = "finin_model0703")

#%%

today_date = sorted(news_grouped.keys())[-1]

today_news = news_grouped[today_date]
today_stock = stock_df.loc[today_date].drop('target')
today_stock = (today_stock - today_stock.mean()) / today_stock.std()

max_news = train_dataset.max_news  # From training dataset
embed_dim = train_dataset.embed_dim
truncated_news = today_news[:max_news]
padded_news = np.zeros((max_news, embed_dim))
padded_news[:len(truncated_news)] = np.stack(truncated_news)

valid_news_length = len(truncated_news) 
news_mask = [False] * valid_news_length + [True] * (max_news - valid_news_length)

news_tensor = torch.tensor(padded_news, dtype=torch.float32).unsqueeze(0)  # Add batch dim
news_mask = torch.tensor(news_mask, dtype=torch.bool).unsqueeze(0)
stock_tensor = torch.tensor(
    today_stock.values.reshape(-1, 5),  # (num_stocks, 5)
    dtype=torch.float32
).unsqueeze(0)

model = torch.load_state_dict(torch.load("finin_model0703.pth", weights_only=True))
model.eval()

with torch.no_grad():
    prediction = model(news_tensor, news_mask, stock_tensor)
    
predicted_change = prediction.item()
print(f"Predicted change for the next day of {today_date} is {predicted_change:.4f}%")
