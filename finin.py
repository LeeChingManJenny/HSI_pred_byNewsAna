# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:49:43 2025

@author: USER
"""
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel

def get_embeddings(texts, batch_size=32):
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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

class StockNewsDataset(Dataset):
    """"
    news_grouped: embadding data of news each day
    stock_data: OLHCV of stock data each day
    news_grouped["Date"] == stock_data["Date"]
    """
    def __init__(self, news_grouped, stock_data,target_mean=None, target_std=None ):
        self.news_grouped = news_grouped
        self.max_news = max(len(v) for v in news_grouped.values())
        self.stock_data = stock_data
        self.dates = sorted(news_grouped.keys())
        #self.max_news_length = max_news_length
        self.embed_dim = len(news_grouped[self.dates[0]][0]) if self.dates else 0 
        #number of embaddings in one row
        
        self.target_mean = target_mean
        self.target_std = target_std
        

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        news = self.news_grouped[date]
        padded = np.zeros((self.max_news, self.embed_dim))
        padded[:len(news)] = np.stack(news)
        mask = [0] * len(news) + [1] * (self.max_news - len(news))
        
        news_emb = torch.tensor(padded, dtype=torch.float32) #convert to tensor formate
        news_mask = torch.tensor(mask, dtype=torch.bool)
        
        value = self.stock_data.loc[date].drop('target').values
        num_news = len(news)
        stock_feat = torch.tensor(value.reshape(-1, 5), dtype=torch.float32)#convert to tensor formate
    
        target = self.stock_data.loc[date]['target']
        if self.target_mean is not None and self.target_std is not None:
            target = (target - self.target_mean) / self.target_std
            
        target = torch.tensor(target, dtype=torch.float32)#convert to tensor formate
        return news_emb,news_mask, stock_feat, target, num_news # X_news, X_stock, y

def collate_fn(batch): # pack everydays data into one
    # Sort by news length
    news_embs, news_masks, stock_feats, targets, num_news = zip(*batch)
    
    #batch.sort(key=lambda x: x['num_news'], reverse=True)
    sorted_indices = sorted(range(len(num_news)), key=lambda i: num_news[i], reverse=True)
    news_embs = [news_embs[i] for i in sorted_indices]
    news_masks = [news_masks[i] for i in sorted_indices]
    stock_feats = [stock_feats[i] for i in sorted_indices]
    targets = [targets[i] for i in sorted_indices]
    
    # Pad news embeddings
    news_padded = torch.stack(news_embs)
    news_masks = torch.stack(news_masks)
    stock_feats = torch.stack(stock_feats)
    targets = torch.stack(targets)
    
    return news_padded, news_masks, stock_feats, targets


""""
"""

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0
        self.stop_training = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True
                
""""
batch_size means number of date

news_padded: (batch_size, max_news, embedding_size = 786)
news_mask: (batch_size, max_news)
stock_input: (batch_size, num_stocks, num_feat = 5)

encoded_news: (batch_size, max_news, 128)
encoded_stock: (batch_size, num_stocks, 128)
cross_att_output: (batch_size, num_stocks, 128)

findal_pred: (batch_size,)

"""
class AttentionModel(nn.Module):
    """"
    news_embed_dim: number of embeding in news_grouped
    stock_feat_dim: number of features in stock_df_ana = len(df_stocks.columns) - 1
    num_heads: Number of Attention Heads, news_embed_dim %% num_heads == 0
    max_news_length: maximum number of array in news_grouped per day, check "size" of news_grouped in each row
    news_emb, stock_feat: get from class StockNewsDataset.__getitem__
    """
    def __init__(self, news_feat=768, stock_feat=5, d_model=128, num_heads=8):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))
        self.aggregation_weight = nn.Linear(d_model, 1)
        
        
        # News Encoder (Self-Attention)
        self.news_proj = nn.Linear(news_feat, d_model)
        self.news_attn = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                activation='gelu',
                batch_first=True
            ),
            num_layers=3  # Add depth
        )
        
        # Stock Encoder (MLP Approach)
        self.stock_encoder = nn.Sequential(
            nn.Linear(stock_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Stock Encoder (LSTM Approach)
        self.stock_proj = nn.Linear(stock_feat, d_model)
        self.stock_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,  # Depth of LSTM
            batch_first=True,
            dropout=0.2  # Optional dropout for multi-layer
        )
        self.stock_norm = nn.LayerNorm(d_model)  # Stabilize LSTM outputs
        
        
        # Cross-Attention (Stock-to-News)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True,  dropout = 0.2
        )
        
        # Predictor
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, news, news_mask, stock):
         # News Encoding: (B, max_news, 300) → (B, max_news, 128)
         news_proj = self.news_proj(news)
         news_encoded = self.news_attn(news_proj, src_key_padding_mask=news_mask)
         
         """
         # Stock Encoding: (B, num_stocks, 5) → (B, num_stocks, 128)
         stock_proj = self.stock_encoder[0](stock)
         stock_encoded = self.stock_encoder[1:](stock_proj) + stock_proj
         """
         # Stock Encoding: LSTM Approach
         stock_proj = self.stock_proj(stock)  # (B, seq_len, d_model)
         stock_lstm_out, _ = self.stock_lstm(stock_proj)  # Process sequence
         stock_encoded = self.stock_norm(stock_lstm_out) + stock_proj  # Residual
         
         # Cross-Attention: Stock → News
         # Query: Stock features (B, num_stocks, 128)
         # Key/Value: News features (B, max_news, 128)
         for layer in [self.news_proj, self.stock_encoder[0]]:
             nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        
         attn_output, _ = self.cross_attn(
            query=stock_encoded,
            key=news_encoded,
            value=news_encoded,
            key_padding_mask=news_mask
         )
         attn_output = self.stock_norm(attn_output + stock_encoded) 
         #attn_output = attn_output + stock_encoded
         
         # Aggregate across stocks (weighted sum)
         aggregation_weights = torch.softmax(self.aggregation_weight(attn_output), dim=1)
         aggregated = (attn_output * aggregation_weights).sum(dim=1)
   
         # Prediction
         output = self.fc(aggregated).squeeze(-1)  # Shape: (B,)
         return output
