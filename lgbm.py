import os
import json
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error


import category_encoders as ce
from sklearn import preprocessing

pd.options.display.max_columns = 999


# In[34]:


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    
    return df


# In[35]:


train_df = pd.read_csv('train_df.csv')



cols_to_remove = ['totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue', 'totals.transactions', 'fullVisitorId', 'visitId']
cols_to_remove = ['totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue', 'totals.transactions', 'fullVisitorId', 'visitId']
train_df.drop(cols_to_remove, axis=1, inplace=True)


# In[ ]:


train_df = add_time_features(train_df)


# ### Train validation split


X_train = train_df[train_df['date']<=datetime.date(2017, 12, 31)]
X_val = train_df[train_df['date']>datetime.date(2017, 12, 31)]

# Get labels
Y_train = X_train['totals.transactionRevenue'].values
Y_val = X_val['totals.transactionRevenue'].values
X_train = X_train.drop(['totals.transactionRevenue'], axis=1)
X_val = X_val.drop(['totals.transactionRevenue'], axis=1)
# Log transform the labels
Y_train = np.log1p(Y_train)
Y_val = np.log1p(Y_val)


# drop date

X_train.drop(['date'], axis=1, inplace=True)
X_val.drop(['date'], axis=1, inplace=True)


features = X_train.columns.values
print('TRAIN SET')
print('Rows: %s' % X_train.shape[0])
print('Columns: %s' % X_train.shape[1])
print('Features: %s' % X_train.columns.values)


# ### Start modelling

params = {
"objective" : "regression",
"metric" : "rmse", 
"num_leaves" : 600,
"min_child_samples" : 20,
"learning_rate" : 0.003,
"bagging_fraction" : 0.6,
"feature_fraction" : 0.7,
"bagging_frequency" : 1,
"bagging_seed" : 1,
"lambda_l1": 3,
'min_data_in_leaf': 50
}


lgb_train = lgb.Dataset(X_train, label=Y_train)
lgb_val = lgb.Dataset(X_val, label=Y_val)
model = lgb.train(params, lgb_train, 1000, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=100, verbose_eval=100)


test_df = pd.read_csv('test_df.csv')
result_df = pd.DataFrame({"fullVisitorId":test_df["fullVisitorId"].values})

test_df.drop(cols_to_remove, axis=1, inplace=True)

test_df = add_time_features(test_df)

y_true = test_df['totals.transactionRevenue']
test_df.drop(['totals.transactionRevenue', 'date'], axis=1, inplace=True)


print('TEST SET')
print('Rows: %s' % test_df.shape[0])
print('Columns: %s' % test_df.shape[1])
print('Features: %s' % test_df.columns.values)

predictions = model.predict(test_df, num_iteration=model.best_iteration)


rms = np.sqrt(mean_squared_error(y_true, predictions))


print(rms)

predictions[predictions<0] = 0

result_df["transactionRevenue"] = y_true.values
result_df["PredictedRevenue"] = np.expm1(predictions)

result_df = result_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(mean_squared_error(np.log1p(result_df["transactionRevenue"].values), np.log1p(result_df["PredictedRevenue"].values))))

