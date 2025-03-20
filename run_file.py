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
import finin
from tqdm import tqdm
#%%
stock_df, news_grouped, dataset,dataloader,stock_feat = news_ana.read_data(update = 0)

train_dataset, val_dataset, train_loader,val_loader, all_loader = news_ana.train_test_split(stock_df, news_grouped)

#%%
model = news_ana.train_finin(train_loader,val_loader,EPOCHS = 100,d_model=64,LR = 1e-7, save = "finin_model200301", plot = 1)#)
