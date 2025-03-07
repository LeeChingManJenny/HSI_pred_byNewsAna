# Stock_ANA
Reference: Wang, M., Cohen, S. B., & Ma, T. (2024). Modeling news interactions and influence for financial market prediction. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2410.10614
## File management:
- web_scr.py: helper function for get_news.py to get news record from futu
- get_news.py: to get 'fina_news.csv.gz' and 'fina_news_senti.csv.gz'
- news_ana.py: convert stock_df and news_df into sutable formate and conduct training of the model
- finin.py: the model
- run_file.py: to run the whole project

## Key field of functions:

### news.ana.py
#### read_data(update = 0, BATCH_SIZE = 16) 
- if update > 0, the 'fina_news.csv.gz' and 'fina_news_senti.csv.gz' will be updated to get most current news
- BATCH_SIZE is to control batch_size of the dataloader
- Return stock_df, news_grouped,dataset,dataloader,stock_feat
- stock_df and news_grouped are the two dataset used in analysis
- dataset = finin.StockNewsDataset(news_grouped, stock_df)

#### train_test_split(stock_df, news_grouped, BATCH_SIZE = 16)
- conduct train-test split for data
- return train_dataset, val_dataset, train_loader,val_loader

#### train_finin(train_loader,val_loader,news_feat=768,d_model=128,num_heads=8,BATCH_SIZE = 16,LR = 2e-4,EPOCHS = 50, save = None)
- train the model
- news_feat=768 is the number of embaddings for each piece of news data
- save = None means do not save the model, or save = "model_name" will save the model as "model_name.pth"


### finin.py
#### get_embeddings(texts, batch_size=32)
- calculate news embeddings by using distilbert-base-uncased model from the DistilBertModel library

#### class StockNewsDataset(Dataset)
- __init__(self, news_grouped, stock_data,target_mean=None, target_std=None )
- __getitem__ return news_emb,news_mask, stock_feat, target, num_news
- news_emb = torch.tensor(<news>, dtype=torch.float32) is the news data in tensor form, with shape (max_news, news_embedding_dim)
- news_mask to handle rather the data in  news_emb is meaningful, with shape (max_news,). For each day, if the number of news < that nax_news, then 0 will be padded to it.
- stock_feat are the Open-High-Low-Close-Volume features for the stock, with shape (num_stocks, 5)
- target is the target to predict.

#### collate_fn(batch)
- pack everydays data into one
- return news_padded, news_masks, stock_feats, targets


#### class AttentionModel(nn.Module)
1. get news_encoded and stock_encoded by passing through news_padded, stock_feats into encoder layer. The shape is transfore to (<Batch>, <max_news>/<stock_feat_num>, <d_model>)
2. get attn_output by 1. passing (news_encoded,stock_encoded) through another cross layer to get news feature 2. adding the stock_encoded feature
3. aggregate news feature and stock feature by weighted sum
4. predict by pass through a sequential model


