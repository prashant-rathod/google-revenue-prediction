
# coding: utf-8

# In[1]:


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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV


import category_encoders as ce
from sklearn import preprocessing

pd.options.display.max_columns = 999


# In[2]:


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    
    return df


# In[3]:


train_df = pd.read_csv('final\\train.csv', dtype={'fullVisitorId': 'str'})


# ## Feature Engineering

# In[4]:


# Impute 0 for missing target values
train_df["totals.transactionRevenue"].fillna(0, inplace=True)

# label encode the categorical variables and convert the numerical variables to float 
# scikit.rf needs numerical data. One hot encoding is not good on rf.
cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']

#these columns should be numbers
num_cols = ["fullVisitorId", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits', 'totals.transactionRevenue']    


#ordinal encoding
encoder = ce.OrdinalEncoder(cols=cat_cols)

train_df = encoder.fit_transform(train_df, train_df["totals.transactionRevenue"])

for col in num_cols:
    train_df[col] = train_df[col].astype(float)


# In[ ]:


cols_to_remove = ['totals.sessionQualityDim', 'totals.timeOnSite', 'totals.totalTransactionRevenue', 'totals.transactions', 'fullVisitorId', 'visitId']
train_df.drop(cols_to_remove, axis=1, inplace=True)


# In[ ]:


train_df.describe()


# In[ ]:


#scaler = preprocessing.StandardScaler()
#scaled_df = scaler.fit_transform(train_df)


# In[ ]:


train_df = add_time_features(train_df)


# ### Train validation split

# In[ ]:


# Get labels
y_train = train_df['totals.transactionRevenue'].values
train_df.drop(['totals.transactionRevenue'], axis=1, inplace=True)
# Log transform the labels
y_train = np.log1p(y_train)


# In[ ]:


# drop date and id columns
train_df.drop(['date'], axis=1, inplace=True)


# In[ ]:


train_df.drop(['visitStartTime','totals.pageviews', 'totals.newVisits', 'totals.bounces'], axis=1, inplace=True)


# In[ ]:


features = train_df.columns.values
print('TRAIN SET')
print('Rows: %s' % train_df.shape[0])
print('Columns: %s' % train_df.shape[1])
print('Features: %s' % train_df.columns.values)


# ### Start modelling

# In[ ]:


tuned_parameters = [{'hidden_layer_sizes': [1,2,3,4,5,6,7,8,9,10,20,30,40],
'activation': ['relu', 'tanh'],
'solver':['adam'], 
'alpha':[0.0001],
'batch_size':['auto'], 'learning_rate':['constant'],
'learning_rate_init':[0.001], 'max_iter':[500],
'momentum': [0.7, 0.5, 0.3]
}]

model = MLPRegressor(activation='relu',solver='adam',
    hidden_layer_sizes=(15,10),
    max_iter=10000,
    shuffle=False,
    random_state=42,
    validation_fraction=0.15,
    momentum=0.7,
    early_stopping=True,)

rgr = GridSearchCV(model, tuned_parameters, cv=5)



# In[ ]:


rgr.fit(train_df, y_train)


# In[ ]:


test_df = pd.read_csv('final\\test.csv', dtype={'fullVisitorId': 'str'})


# In[ ]:


#apply encoing
test_df = encoder.transform(test_df)

for col in num_cols:
    test_df[col] = test_df[col].astype(float)


# In[ ]:


# extract fullVisitorId before removing it


result_df = pd.DataFrame({"fullVisitorId":test_df["fullVisitorId"].values})


# In[ ]:


test_df = add_time_features(test_df)


# In[ ]:


test_df.drop(['visitStartTime','totals.pageviews', 'totals.newVisits', 'totals.bounces'], axis=1, inplace=True)


# In[ ]:


y_true = test_df['totals.transactionRevenue']

additional_cols_remove = ['totals.transactionRevenue', 'date', 'fullVisitorId']
# drop
test_df.drop(cols_to_remove + additional_cols_remove, axis=1, inplace=True)


# In[ ]:


print('TEST SET')
print('Rows: %s' % test_df.shape[0])
print('Columns: %s' % test_df.shape[1])
print('Features: %s' % test_df.columns.values)


# In[ ]:


predictions = rgr.predict(test_df)

print(rgr.best_params_)
print(rgr.best_score_)

# In[ ]:


rms = np.sqrt(mean_squared_error(y_true, predictions))


# In[ ]:


print('rmse', rms)


predictions[predictions<0] = 0
result_df["transactionRevenue"] = y_true.values
result_df["PredictedRevenue"] = np.expm1(predictions)

result_df = result_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
final_error = np.sqrt(mean_squared_error(np.log1p(result_df["transactionRevenue"].values), np.log1p(result_df["PredictedRevenue"].values)))
print('final_error', final_error)

