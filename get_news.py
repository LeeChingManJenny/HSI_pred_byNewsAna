# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:16:29 2025

@author: USER
"""

def main(n = 10):
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    import web_scr as ws
    import pandas as pd
    #%%
    news_df = pd.read_csv('fina_news.csv.gz', compression='gzip')
    driver = ws.scroll(n) #%% update news data
    
    news_df = ws.get_news(driver,news_df)#
    
    news_df.to_csv('fina_news.csv.gz', index=False, compression='gzip')
    
    #%% get sentiment score
    news_df = pd.read_csv('fina_news.csv.gz', compression='gzip')
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def get_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs) 
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.detach().numpy()[0]
    
    #%%
    senti = []
    j=0
    for i in news_df['Header']:
        print(j)
        senti.append(get_sentiment(i))
        j=j+1
        
    #%%
    
    senti_df = pd.DataFrame(senti, columns=["Negative", "Neutral", "Positive"])
    news_df = pd.concat([news_df, senti_df], axis=1)
    
    news_df.to_csv('fina_news_senti.csv.gz', index=False, compression='gzip')
    
    return