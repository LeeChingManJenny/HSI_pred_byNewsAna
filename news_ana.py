# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:37:05 2025

@author: USER
"""


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import numpy as np 
import pandas as pd
import stock_fun as sf
import get_news 
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import finin
import torch.nn as nn
from torch.utils.data import DataLoader
#%%
#find news embaddings
get_news.main(10) #update news 
#%%
news_df = pd.read_csv('fina_news_senti.csv.gz', compression='gzip')
emb_news_df = pd.read_csv('fina_news_senti_emb.csv.gz', compression='gzip')

#%%

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_embeddings(texts, batch_size=32):
    model.eval()
    embeddings = []
    
    # Process in batches to handle large datasets
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,  # Adjust based on headline length
            return_tensors="pt"
        ).to(device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use [CLS] token embeddings as sentence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(cls_embeddings)
    
    return embeddings

"""
headlines = []

for i in news_df["Header"]:
    if i not in emb_news_df["Header"]:
        headlines = np.append(headlines,i)
#headlines = news_df["Header"].tolist()
embeddings = get_embeddings(headlines)
embedding_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings[0].shape[0])])
emb_news_df = pd.concat([
    news_df[["Date","Header", "Negative", "Neutral", "Positive"]].reset_index(drop=True),
    embedding_df
], axis=1)

"""

mask = ~news_df["Header"].isin(emb_news_df["Header"])
new_entries = news_df[mask]

if not new_entries.empty:
    # Calculate embeddings only for new headlines
    new_headlines = new_entries["Header"].tolist()
    embeddings = get_embeddings(new_headlines)
    
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
#%%
# the data use to analys is news_grouped and stock_df_ana, while stock_df can be use to predict tmr
td = datetime.today().strftime('%Y-%m-%d')
hsi_data = sf.get_yfdata('HSI','2015-01-05',endd =td )
ixic_data = sf.get_yfdata('IXIC','2015-01-05')
n225_data = sf.get_yfdata('N225','2015-01-05')
gdaxi_data = sf.get_yfdata('GDAXI','2015-01-05')

comm_data = sf.merge_df('Date','left',1,hsi_data,ixic_data,n225_data,gdaxi_data)

stock_df = comm_data

embed_cols = [col for col in emb_news_df.columns if col.startswith('emb_')]

emb_news_df['emb'] = emb_news_df[embed_cols].apply(lambda x: x.values, axis=1) # group all embeding into one column

news_by_day = emb_news_df.groupby('Date')['emb'].apply(list).to_dict()

stock_df['target'] = stock_df['HSI4'].pct_change().shift(-1)

valid_dates = set(news_by_day.keys()) & set(stock_df.index.strftime('%Y-%m-%d'))
stock_df = stock_df[stock_df.index.isin(valid_dates)]
news_grouped = {date: news_by_day[date] for date in valid_dates}
stock_df_ana = stock_df.dropna()

sorted_dates = sorted(news_grouped.keys())  # Sort keys as strings
news_grouped = {date: news_grouped[date] for date in sorted_dates}

news_grouped_ana = news_grouped
a = news_grouped_ana.popitem()
#%% Process data to sutable form for modeling
""""
news_group: dict
Key: Date, Row type: list of array
Each row store the news embadding of each day

stock_data = stock_df_ana: dataframe
Each row store OLHCV of multiple stocks, and target = (HSI_tmr - HSI_td)/HSI_td
"""

"""
## to process data

finin.StockNewsDataset(Dataset)
news_grouped: embadding data of news each day
stock_data: OLHCV of stock data each day
news_grouped["Date"] == stock_data["Date"]


## the cross attention model

finin.AttentionModel(nn.Module)
news_embed_dim: number of embeding in news_grouped
stock_feat_dim: number of features in stock_df_ana = len(df_stocks.columns) - 1
num_heads: Number of Attention Heads, news_embed_dim %% num_heads == 0
max_news_length: maximum number of array in news_grouped per day, check "size" of news_grouped in each row
news_emb, stock_feat: get from class StockNewsDataset.__getitem__
"""

#%%
EMBED_DIM = 768  # Match your embedding dimension
STOCK_FEAT_DIM = len(stock_df_ana.columns) - 1  # Exclude 'target'
NUM_HEADS = 4
MAX_NEWS = 50
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10

def collate_fn(batch):
    news, stocks, targets = zip(*batch)
    return (
        torch.stack(news),  # (batch_size, seq_len, embed_dim)
        torch.stack(stocks),  # (batch_size, stock_feat_dim)
        torch.stack(targets)  # (batch_size,)
    )

dataset = finin.StockNewsDataset(news_grouped_ana, stock_df_ana, MAX_NEWS)
dataset2 = finin.StockNewsDataset(news_grouped, stock_df, MAX_NEWS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn)
#%%

model = finin.AttentionModel(EMBED_DIM, STOCK_FEAT_DIM, NUM_HEADS, MAX_NEWS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for news, stocks, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(news, stocks)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
#%%

model.eval()
with torch.no_grad():
    sample_news, sample_stock, _ = dataset[0]
    prediction = model(sample_news.unsqueeze(1), sample_stock)
    print(f'Predicted % change: {prediction.item():.4f}')