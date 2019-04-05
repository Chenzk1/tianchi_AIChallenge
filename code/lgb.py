#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import warnings
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm.plotting import plot_importance
from lightgbm import LGBMRegressor
from scipy import sparse
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import seaborn as sns

sns.set()
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',200)


# In[2]:


BASE_PATH = os.path.join('../input')
RAW_PATH = os.path.join(BASE_PATH, 'RAW_DATA')
TRAIN_PATH = os.path.join(RAW_PATH, 'Metro_train')
TEST_A_PATH = os.path.join(RAW_PATH, 'Metro_testA')
SUBMIT_PATH = os.path.join('../submit')


# In[4]:


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props
def read_data(name, **params):
    data = pd.read_csv(name, **params)
    data = reduce_mem_usage(data)
    return data


# ## 读取数据
# payType-most
# devideID-most two
# userID-count(this stationID)-( nunique(userID) - nunique(userID)[payType==3])
def get_hour_cut(data):
    if data>= 23 or data <= 6:
        hour_cut = 1
    elif data>= 10 and data <= 13:
        hour_cut = 2
    elif data>= 18 and data <= 22:
        hour_cut = 3
    elif data>= 14 and data <= 17:
        hour_cut = 4
    else:
        hour_cut = 5
    return hour_cut
def is_weekend(data):
    if data <= 4:
        return 0
    else:
        return 1
def date_processing(data):
    data['startTime'] = data['time'].apply(lambda x: str(x)[:15]+ '0:00')
    data['day'] = data['startTime'].apply(lambda x: int(str(x)[8:10]))
    data['hour'] = data['startTime'].apply(lambda x: int(str(x)[11:13]))
    data['minute'] = data['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
    data['startTime'] = pd.to_datetime(data['startTime'],format= '%Y-%m-%d %H:%M:%S')
    data['weekday'] = data['startTime'].dt.weekday
    #result['weekend'] = result['weekday'].apply(is_weekend)
    
    result = data.groupby(['stationID', 'startTime','day', 'hour', 'minute','weekday'])['status'].agg(['count','sum'])
    result = result.reset_index()
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']
    
    tmp     = data.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result  = result.merge(tmp, on=['stationID'], how='left')
    tmp     = data.groupby(['stationID','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result  = result.merge(tmp, on=['stationID','hour'], how='left')
    tmp     = data.groupby(['stationID','hour','minute'])['deviceID'].nunique().                                           reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result  = result.merge(tmp, on=['stationID','hour','minute'], how='left')
    def get_top(df, n=1):
        return df.sort_values()[-n:].values[0]
    tmp     = data.groupby(['stationID'])['deviceID'].apply(get_top,n=1).reset_index(name='most_deviceID_of_stationID')
    result  = result.merge(tmp, on=['stationID'], how='left')

    tmp     = data.groupby(['stationID','hour'])['deviceID'].apply(get_top,n=1).reset_index(name='most_deviceID_of_stationID&hour')
    result  = result.merge(tmp, on=['stationID','hour'], how='left')

    tmp     = data.groupby(['stationID','weekday','hour'])['deviceID'].apply(get_top,n=1).reset_index(name='most_deviceID_of_stationID&wh')
    result  = result.merge(tmp, on=['stationID','weekday','hour'], how='left')

    tmp     = data.groupby(['stationID'])['payType'].apply(get_top,n=1).reset_index(name='most_payType_of_stationID')
    result  = result.merge(tmp, on=['stationID'], how='left')
    tmp     = data.groupby(['stationID','hour'])['payType'].apply(get_top,n=1).reset_index(name='most_payType_of_stationID&hour')
    result  = result.merge(tmp, on=['stationID','hour'], how='left')
    tmp     = data.groupby(['stationID','weekday','hour'])['payType'].apply(get_top,n=1).reset_index(name='most_payType_of_stationID&wh')
    result  = result.merge(tmp, on=['stationID','weekday','hour'], how='left')

    #result['weekday'] = result['startTime'].dt.weekday
    result['hourCut'] = result['hour'].map(get_hour_cut)
    result = result.drop(columns=['count', 'sum'])
    # datetime -> int
    return result
def date_processing_test(data):
    result = data
    
    result['day'] = result['startTime'].apply(lambda x: int(str(x)[8:10]))
    result['startTime'] = result['time'].apply(lambda x: str(x)[:15]+ '0:00')
    result['startTime'] = pd.to_datetime(result['startTime'],format= '%Y-%m-%d %H:%M:%S')
    result['weekday'] = result['startTime'].dt.weekday
    #result['weekend'] = result['weekday'].apply(is_weekend)
    result['hour'] = result['startTime'].apply(lambda x: int(str(x)[11:13]))
    result['minute'] = result['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
    result['hourCut'] = result['hour'].map(get_hour_cut)
    result = result.drop(columns='endTime')
    result = result.drop(columns=['count', 'sum'])
    return result

    

## 读取数据
data = pd.DataFrame()
for file in os.listdir(TRAIN_PATH):
    print(f'the file: {file}')
    temp = read_data(os.path.join(TRAIN_PATH, file))
    temp = date_processing(temp)
    data = pd.concat([data, temp],ignore_index=True)
    del temp
test_name = os.path.join(TEST_A_PATH, 'testA_record_2019-01-28.csv')
test_28 = read_data(test_name)
test_28 = date_processing(test_28)
data = pd.concat([data, test_28],ignore_index=True)
test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
test = pd.read_csv(test_name)
test = date_processing_test(test)


## F E
# 剔除周末,并修改为连续时间
data = data[(data.day!=5)&(data.day!=6)]
data = data[(data.day!=12)&(data.day!=13)]
data = data[(data.day!=19)&(data.day!=20)]
data = data[(data.day!=26)&(data.day!=27)]

def fix_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [7,8,9,10,11]:
        return d - 2
    elif d in [14,15,16,17,18]:
        return d - 4
    elif d in [21,22,23,24,25]:
        return d - 6
    elif d in [28]:
        return d - 8
data['day'] = data['day'].apply(fix_day)


# In[27]:


#test = test.drop(['startTime'], axis=1)
data = pd.concat([data,test], axis=0, ignore_index=True)

stat_columns = ['inNums','outNums']
## 特征构造

def get_refer_day(d):
    if d == 20:
        return 29
    else:
        return d + 1
# 1->1 1->2 2->3...20->29
tmp = data.copy()
tmp_df = tmp[tmp.day==1]
tmp_df['day'] = tmp_df['day'] - 1
tmp = pd.concat([tmp, tmp_df], axis=0, ignore_index=True)
tmp['day'] = tmp['day'].apply(get_refer_day)


for f in stat_columns:
    tmp.rename(columns={f: f+'_last'}, inplace=True) 
    
tmp = tmp[['stationID','day','hour','minute','inNums_last','outNums_last']]
# 相当于把前一天的innum和outnum加在了当天行
# 但是要求比较严格 必须是同十分钟内的
data = data.merge(tmp, on=['stationID','day','hour','minute'], how='left')
data.fillna(0, inplace=True)

tmp = data.copy()
tmp.set_index(["startTime"], inplace=True)

# In[29]:


tmp = data.groupby(['stationID','weekday','hour','minute'], as_index=False)['inNums'].agg({
                                                                        'inNums_whm_max'    : 'max',
                                                                        'inNums_whm_min'    : 'min',
                                                                        'inNums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

tmp = data.groupby(['stationID','weekday','hour','minute'], as_index=False)['outNums'].agg({
                                                                        'outNums_whm_max'    : 'max',
                                                                        'outNums_whm_min'    : 'min',
                                                                        'outNums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

tmp = data.groupby(['stationID','weekday','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','weekday','hour'], how='left')

tmp = data.groupby(['stationID','weekday','hour'], as_index=False)['outNums'].agg({
                                                                        #'outNums_wh_max'    : 'max',
                                                                        #'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','weekday','hour'], how='left')


# In[36]:


def recover_day(d):
    if d in [1,2,3,4]:
        return d
    elif d in [5,6,7,8,9]:
        return d + 2
    elif d in [10,11,12,13,14]:
        return d + 4
    elif d in [15,16,17,18,19]:
        return d + 6
    elif d == 20:
        return d + 8
    else:
        return d
    
data = data.drop(columns='startTime')
all_columns = [f for f in data.columns if f not in ['weekend','inNums','outNums']]
### all data
all_data = data[data.day!=29]
all_data['day'] = all_data['day'].apply(recover_day)
X_data = all_data[all_columns].values

train = data[data.day <20]
train['day'] = train['day'].apply(recover_day)
X_train = train[all_columns].values

valid = data[data.day==20]
valid['day'] = valid['day'].apply(recover_day)
X_valid = valid[all_columns].values

test  = data[data.day==29]
X_test = test[all_columns].values


# In[37]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':1,
    'reg_lambda':2
}

######################################################inNums
y_train = train['inNums']
y_valid = valid['inNums']
y_data  = all_data['inNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100000,
                valid_sets=[lgb_train,lgb_evals],
                valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['inNums'] = gbm.predict(X_test)

######################################################outNums
y_train = train['outNums']
y_valid = valid['outNums']
y_data  = all_data['outNums']
lgb_train = lgb.Dataset(X_train, y_train)
lgb_evals = lgb.Dataset(X_valid, y_valid , reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=[lgb_train,lgb_evals],
                #valid_names=['train','valid'],
                early_stopping_rounds=200,
                verbose_eval=1000,
                )

### all_data
lgb_train = lgb.Dataset(X_data, y_data)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=gbm.best_iteration,
                valid_sets=[lgb_train],
                valid_names=['train'],
                verbose_eval=1000,
                )
test['outNums'] = gbm.predict(X_test)

sub = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
sub['inNums']   = test['inNums'].values
sub['outNums']  = test['outNums'].values
# 结果修正
sub.loc[sub.inNums<0 , 'inNums']  = 0
sub.loc[sub.outNums<0, 'outNums'] = 0
sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('output/sub_model.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


X_train = train[['stationID', 'date', 'startTime', 'weekday','hourCut']]
y_train_1 = train['inNums']
y_train_2 = train['outNums']
X_test_28 = test_28[['stationID', 'date', 'startTime', 'weekday','hourCut']]
y_test_28_1 = test_28['inNums']
y_test_28_2 = test_28['outNums']
X_test_29 = test_29[['stationID', 'date', 'startTime', 'weekday','hourCut']]
params = {
    'bagging_freq': 10,          
    'bagging_fraction': 0.3,   'boost_from_average':'false',   
    'boost': 'gbdt',             
    #'feature_fraction': 0.0405,     
    'learning_rate': 0.1,
    'max_depth': -1,             'metric':'mae',                
    'min_data_in_leaf': 80, 
    'num_leaves': 13,            
    'num_threads': -1, 
    'objective': 'regression_l1',       'verbosity': 1,
    'num_boost_round': 10000000
}
NFOLD = 15
folds = KFold(n_splits=NFOLD, random_state=134, shuffle=True)
val_lgb1 = np.zeros(len(X_train))
pred_lgb1 = np.zeros(len(X_test_29))
pred_28_1 = np.zeros(len(X_test_28))
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train_1)):
    print(f'fold: {n_fold}')
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], y_train_1[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], y_train_1[val_idx])
    
    reg_lgb1 = lgb.train(params, trn_data, num_boost_round=2000000, valid_sets=[trn_data, val_data], verbose_eval=10000, early_stopping_rounds=600)
    #val_lgb1[val_idx] = reg_lgb1.predict(X_train.iloc[val_idx], num_iteration=reg_lgb.best_iteration)
    pred_lgb1 += reg_lgb1.predict(X_test_29, num_iteration=reg_lgb1.best_iteration) / NFOLD
    pred_28_1 += reg_lgb1.predict(X_test_28, num_iteration=reg_lgb1.best_iteration) / NFOLD 
print(f'mae error: {mean_absolute_error(pred_28_1, y_test_28_1)}')

folds = KFold(n_splits=NFOLD, random_state=134, shuffle=True)
val_lgb2 = np.zeros(len(X_train))
pred_lgb2 = np.zeros(len(X_test_29))
pred_28_2 = np.zeros(len(X_test_28))
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train_2)):
    print(f'fold: {n_fold}')
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], y_train_2[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], y_train_2[val_idx])
    
    reg_lgb2 = lgb.train(params, trn_data, num_boost_round=2000000, valid_sets=[trn_data, val_data], verbose_eval=10000, early_stopping_rounds=600)
    #val_lgb2[val_idx] = reg_lgb2.predict(X_train.iloc[val_idx], num_iteration=reg_lgb.best_iteration)
    pred_lgb2 += reg_lgb2.predict(X_test_29, num_iteration=reg_lgb2.best_iteration) / NFOLD
    pred_28_2 += reg_lgb2.predict(X_test_28, num_iteration=reg_lgb2.best_iteration) / NFOLD 
print(f'mae error: {mean_absolute_error(pred_28_2, y_test_28_2)}')


# In[73]:


folds = KFold(n_splits=NFOLD, random_state=134, shuffle=True)
val_lgb2 = np.zeros(len(X_train))
pred_lgb2 = np.zeros(len(X_test_29))
pred_28_2 = np.zeros(len(X_test_28))
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train_2)):
    print(f'fold: {n_fold}')
    trn_data = lgb.Dataset(X_train.iloc[trn_idx], y_train_2[trn_idx])
    val_data = lgb.Dataset(X_train.iloc[val_idx], y_train_2[val_idx])
    
    reg_lgb2 = lgb.train(params, trn_data, num_boost_round=2000000, valid_sets=[trn_data, val_data], verbose_eval=10000, early_stopping_rounds=600)
    #val_lgb2[val_idx] = reg_lgb2.predict(X_train.iloc[val_idx], num_iteration=reg_lgb.best_iteration)
    pred_lgb2 += reg_lgb2.predict(X_test_29, num_iteration=reg_lgb2.best_iteration) / NFOLD
    pred_28_2 += reg_lgb2.predict(X_test_28, num_iteration=reg_lgb2.best_iteration) / NFOLD 
print(f'mae error: {mean_absolute_error(pred_28_2, y_test_28_2)}')


# In[74]:


test_29['inNums'] = pred_lgb1
test_29['outNums'] = pred_lgb2


# In[75]:


submit_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
submit = pd.read_csv(submit_name) 
test_29['startTime'] = submit['startTime']
test_29[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(SUBMIT_PATH+'/lgb.csv', index=False)

