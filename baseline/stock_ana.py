 # -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:41:58 2025

@author: USER
"""

import numpy as np 
import pandas as pd
import os

#%%

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import stock_fun as sf
from datetime import datetime

td = datetime.today().strftime('%Y-%m-%d')
hsi_data = sf.get_yfdata('HSI','2015-01-05',endd =td )
ixic_data = sf.get_yfdata('IXIC','2015-01-05')
n225_data = sf.get_yfdata('N225','2015-01-05')
gdaxi_data = sf.get_yfdata('GDAXI','2015-01-05')
#fchi_data = sf.get_yfdata('FCHI','2015-01-05') 
#nya_data = sf.get_yfdata('NYA','2015-01-05')
#dji_data = sf.get_yfdata('DJI','2015-01-05')

#join the above dataset into one dataframe by left joining Date
comm_data = sf.merge_df('Date','left',1,hsi_data,ixic_data,n225_data,gdaxi_data)#fchi_data,nya_data,dji_data)

pro_data1 = sf.shift_data(comm_data,3,'HSI4')
#%% Variable selection
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNetCV
from time import time
from sklearn.feature_selection import SequentialFeatureSelector

X = pro_data1.iloc[:,1:]
y = pro_data1.iloc[:,0]
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=60)).fit(X, y)
sgd = SGDRegressor(max_iter=1000, tol=1e-3).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(X.columns)
plt.bar(height=importance, x=feature_names)
plt.title("Feature importances via coefficients")
plt.show()

tmp = pd.DataFrame({'name': X.columns, 'coef': importance})

#%%


tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=5, direction="forward"
).fit(X, y)
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=3, direction="forward"
).fit(X, y)
toc_bwd = time()


#%%
print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")
print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")

aic = feature_names[sfs_forward.get_support()]
bic = feature_names[sfs_backward.get_support()]

sel_var = list(set(aic) | set(bic))
#sel_var = list(set(aic))
pro_data = pro_data1[sel_var]
pro_data.insert(0, 'y', pro_data1['y'])
#%%linear regression model 0.70
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

train,test = sf.split_train_test(pro_data)
lr = make_pipeline(StandardScaler(),
                    linear_model.LinearRegression())

x = pro_data.iloc[:,1:]
y = pro_data.iloc[:,0]
x_train = train.iloc[:,1:]
x_test = test.iloc[:,1:]
y_train = train.iloc[:,0]
y_test = test.iloc[:,0]

lr.fit(x_train, y_train)

train_pred = lr.predict(x_train)
test_pred = lr.predict(x_test)

# Display the DataFrame
sf.displayMAPE(sf.calMape(y_train,train_pred),'LR',0)
sf.displayMAPE(sf.calMape(y_test,test_pred),'LR',1)
#%% Smoothing 1.17
test_pred2 = sf.exp_smooth(y_test, a = 1)
sf.calMape(y_test,test_pred2)

#%%SGD 0.63



reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))

reg.fit(x_train, y_train)

pred1 = reg.predict(x_train)
pred2 = reg.predict(x_test)

sf.displayMAPE(sf.calMape(y_train,pred1),'SGD',0)
sf.displayMAPE(sf.calMape(y_test,pred2),'SGD',1)
#%%E-net


enet = ElasticNetCV(cv=3, random_state=22)
enet.fit(x_train, y_train)

pred_E1 = enet.predict(x_train)
pred_E2 = enet.predict(x_test)

sf.displayMAPE(sf.calMape(y_train,pred_E1),'E-Net',0)
sf.displayMAPE(sf.calMape(y_test,pred_E2),'E-Net',1)
#%%

plt.figure(figsize=(10, 5))

plt.plot(test_pred, label='Test Prediction', color='blue')

#plt.plot(test_pred2, label='Test Prediction Smooth', color='red', marker='o')

plt.plot(y_test, label='Actual Close', color='orange')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Test Predictions vs Actual Close Prices')
plt.legend()

# Show the plot
plt.grid()
plt.show()


#%%
import warnings
warnings.filterwarnings("ignore")

lr.fit(x, y)
reg.fit(x, y)

today1 = sf.oneday_predict(comm_data,3,2)
tmr1 = sf.oneday_predict(comm_data,3,1)

today = today1[sel_var]
tmr = tmr1[sel_var]

td = reg.predict(today)
tm = reg.predict(tmr)

td = td.item()
tm= tm.item()
ratio = ((tm-td)/td)*100

print(f"today: {td:.2f} tomorrow: {tm:.2f}; predicted ups/downs: {ratio:.2f}%")

#%%

import stock_ana1 as an
an.main()