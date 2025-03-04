# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:49:43 2025

@author: USER
"""
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class StockNewsDataset(Dataset):
    """"
    news_grouped: embadding data of news each day
    stock_data: OLHCV of stock data each day
    news_grouped["Date"] == stock_data["Date"]
    """
    def __init__(self, news_grouped, stock_data):
        self.news_grouped = news_grouped
        self.dates = list(news_grouped.keys()) # date appear in news_group
        self.stock_data = stock_data
        #self.max_news_length = max_news_length
        self.embed_dim = len(news_grouped[self.dates[0]][0]) if self.dates else 0 #number of embaddings in one row

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        news_emb = torch.tensor(self.news_grouped[date], dtype=torch.float32) #convert to tensor formate
        # Pad news embeddings
        if news_emb.shape[0] < self.max_news_length:
            pad = torch.zeros(self.max_news_length - len(news_emb), self.embed_dim)
            news_emb = torch.cat([news_emb, pad])
        else:
            news_emb = news_emb[:self.max_news_length]
        
        stock_feat = torch.tensor(self.stock_data.loc[date].drop('target'), dtype=torch.float32).squeeze()#convert to tensor formate
        target = torch.tensor(self.stock_data.loc[date]['target'], dtype=torch.float32)#convert to tensor formate
        return news_emb, stock_feat, target # X_news, X_stock, y

def collate_fn(batch):
    # Sort by news length
    batch.sort(key=lambda x: x['num_news'], reverse=True)
    
    # Pad news embeddings
    news_padded = pad_sequence(
        [item['news_embeddings'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    # Create news masks
    news_masks = torch.stack([
        torch.cat([
            torch.ones(item['num_news']),
            torch.zeros(news_padded.size(1) - item['num_news'])
        ]) for item in batch
    ]).bool()
    
    # Stack stock features (batch_size, num_stocks, 5)
    stock_features = torch.stack([item['stock_features'] for item in batch])
    
    return {
        'news_embeddings': news_padded,
        'news_masks': news_masks,
        'stock_features': stock_features
    }
   
class AttentionModel(nn.Module):
    """"
    news_embed_dim: number of embeding in news_grouped
    stock_feat_dim: number of features in stock_df_ana = len(df_stocks.columns) - 1
    num_heads: Number of Attention Heads, news_embed_dim %% num_heads == 0
    max_news_length: maximum number of array in news_grouped per day, check "size" of news_grouped in each row
    news_emb, stock_feat: get from class StockNewsDataset.__getitem__
    """
    def __init__(self, news_embed_dim, stock_feat_dim, num_heads):
        super().__init__()
        #self.max_news_length = max_news_length #50
        
        # Self-attention layer
        # = news_embed_dim* num_heads
        self.self_attn = nn.MultiheadAttention(news_embed_dim, num_heads)
        
        # Project stock features to query
        self.stock_proj = nn.Linear(stock_feat_dim, news_embed_dim)
        
        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(news_embed_dim, num_heads)
        
        # Prediction layers
        self.fc = nn.Sequential(
            nn.Linear(news_embed_dim + stock_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, news_emb, stock_feat):
        # news_emb: (seq_len, batch=1, embed_dim)
        batch_size = news_emb.size(0)
        news_emb = news_emb.permute(1, 0, 2)  # (seq_len, 1, embed_dim)
        
        # Self-attention
        self_attn_out, _ = self.self_attn(news_emb, news_emb, news_emb)
        
        # Cross-attention (query from stock, key/value from news)
        q = self.stock_proj(stock_feat).unsqueeze(0)
        q = q.permute(1, 0, 2)
        #q = self.stock_proj(stock_feat).unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)
        cross_attn_out, _ = self.cross_attn(q, self_attn_out, self_attn_out)
        ad = cross_attn_out.squeeze(0)
        
        # Concatenate with stock features
        combined = torch.cat([ad, stock_feat], dim=1)
        return self.fc(combined)