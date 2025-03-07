# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:37:05 2025

@author: USER
"""


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import pandas as pd
import stock_fun as sf
import get_news 
import torch
import finin_prob as finin
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

#%%
def read_data(update = 0, BATCH_SIZE = 16):
    
    if update != 0:
        #find news embaddings
        get_news.main(update) #update fina_news_senti.csv.gz
        
    news_df = pd.read_csv('fina_news_senti.csv.gz', compression='gzip')
    emb_news_df = pd.read_csv('fina_news_senti_emb.csv.gz', compression='gzip')
    
    mask = ~news_df["Header"].isin(emb_news_df["Header"])
    new_entries = news_df[mask]
    
    if not new_entries.empty:
        # Calculate embeddings only for new headlines
        new_headlines = new_entries["Header"].tolist()
        embeddings = finin.get_embeddings(new_headlines)
        
        # Create DataFrame for new embeddings
        embedding_columns = [f"emb_{i}" for i in range(len(embeddings[0]))]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)
        
        # Combine new entries with their embeddings
        new_data = pd.concat([
            new_entries[["Date", "Header", "Negative", "Neutral", "Positive"]].reset_index(drop=True),
            embedding_df
        ], axis=1)
        
        # Append to existing emb_news_df
        emb_news_df = pd.concat([emb_news_df, new_data], ignore_index=True)
        
    emb_news_df.to_csv('fina_news_senti_emb.csv.gz', index=False, compression='gzip')
    
    emb_news_df["Date"] = pd.to_datetime(emb_news_df["Date"], format="%Y-%m-%d")
    emb_news_df["Date"] = emb_news_df["Date"] - pd.Timedelta(days=1)
    emb_news_df["Date"] = emb_news_df["Date"].dt.strftime("%Y-%m-%d")
    
    # the data use to analys is news_grouped and stock_df_ana, while stock_df can be use to predict tmr
    hsi_data = sf.get_yfdata('HSI','2015-01-05')
    ixic_data = sf.get_yfdata('IXIC','2015-01-05')
    n225_data = sf.get_yfdata('N225','2015-01-05')
    gdaxi_data = sf.get_yfdata('GDAXI','2015-01-05')
    
    comm_data = sf.merge_df('Date','left',1,hsi_data,ixic_data,n225_data,gdaxi_data)
    
    stock_df = comm_data
    
    embed_cols = [col for col in emb_news_df.columns if col.startswith('emb_')]
    
    
    emb_news_df['emb'] = emb_news_df[embed_cols].apply(lambda x: x.values, axis=1) # group all embeding into one column
    
    news_by_day = emb_news_df.groupby('Date')['emb'].apply(list).to_dict()
    
    tmp = stock_df['HSI4'].pct_change().shift(-1)*100
    
    stock_df['target'] = (tmp > 0).astype(int)
    
    valid_dates = set(news_by_day.keys()) & set(stock_df.index.strftime('%Y-%m-%d'))
    stock_df = stock_df[stock_df.index.isin(valid_dates)]
    news_grouped = {date: news_by_day[date] for date in valid_dates}
    
    stock_df_ana = stock_df.dropna()
     
    sorted_dates = sorted(news_grouped.keys())  # Sort keys as strings
    news_grouped_ana = {date: news_grouped[date] for date in sorted_dates}
    news_grouped_ana.popitem()
    
    dataset = finin.StockNewsDataset(news_grouped_ana, stock_df_ana)
    dataloader = DataLoader(
        dataset, 
        batch_size= BATCH_SIZE, 
        collate_fn= finin.collate_fn, 
        shuffle=False
    )
    
    stock_feat=len(stock_df_ana.columns) - 1

    return stock_df, news_grouped,dataset,dataloader,stock_feat


#%%Train-test split


def train_test_split(stock_df, news_grouped, BATCH_SIZE = 16):
    
    stock_df_ana = stock_df.dropna()
    sorted_dates = sorted(news_grouped.keys())  # Sort keys as strings
    news_grouped_ana = {date: news_grouped[date] for date in sorted_dates}
    news_grouped_ana.popitem()
    
    dates = sorted(news_grouped_ana.keys())
    split_idx = int(0.8 * len(dates))  
    train_dates = dates[:split_idx]
    train_news = {date: news_grouped_ana[date] for date in train_dates}
    train_stock = stock_df_ana.loc[train_dates]
    train_stock = (train_stock - train_stock.mean()) / train_stock.std()
    
    train_targets = train_stock['target'].values
    #target_mean = train_targets.mean()
    #target_std = train_targets.std()
    
    val_dates = dates[split_idx:]
    val_news = {date: news_grouped_ana[date] for date in val_dates}
    val_stock = stock_df_ana.loc[val_dates]
    val_stock = (val_stock - val_stock.mean()) / val_stock.std()
    
    
    train_dataset = finin.StockNewsDataset(train_news, train_stock)
    val_dataset = finin.StockNewsDataset(val_news, val_stock)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=finin.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=finin.collate_fn)
    
    return train_dataset, val_dataset, train_loader,val_loader
#%%
####################################

def train_finin(train_loader,val_loader,news_feat=768,d_model=128,num_heads=8,BATCH_SIZE = 16,LR = 2e-4,EPOCHS = 50, save = None):
    gc.collect()
    torch.cuda.empty_cache()
    
    model = finin.AttentionModel(news_feat=news_feat, stock_feat=5, d_model=d_model, num_heads=num_heads)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=2e-4,
        total_steps= EPOCHS * len(train_loader),
        pct_start=0.1
    )
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0 
        for news_padded, news_masks, stock_feats, targets in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(news_padded, news_masks, stock_feats)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')
        
        model.eval()
    
        val_loss = 0.0
    
        with torch.no_grad():
            for news_padded, news_masks, stock_feats, targets in tqdm(val_loader):
                preds = model(news_padded, news_masks, stock_feats)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                print(preds.item)
                print(f'loss: {loss:.4f}')
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if save is not None:
            torch.save(model.state_dict(), f"{save}.pth") 
        

#%%
