#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import norm, rankdata
import warnings
import gc
import os
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics

plt.style.use('seaborn')
sns.set(font_scale=1)
pd.set_option('display.max_columns', 500)
#%%
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

path = '../input/RAW_DATA'
test = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
test_28 = read_data(path + '/Metro_testA/testA_record_2019-01-28.csv')

def get_hour_cut(data):
    if data>= 22 or data <= 6:
        hour_cut = 1
    elif data>= 10 and data <= 15:
        hour_cut = 2
    elif data>= 16 and data <= 18:
        hour_cut = 4
    elif (data>= 19 and data <= 21) or data==7:
        hour_cut = 3
    else:
        hour_cut = 5
    return hour_cut
def get_base_features(df_):
    '''
    day week weekend hour minute
    '''
    df = df_.copy()
    
    # base time
    df['day']     = df['time'].apply(lambda x: int(x[8:10]))
    df['week']    = pd.to_datetime(df['time']).dt.dayofweek + 1
    df['weekend'] = (pd.to_datetime(df.time).dt.weekday >=5).astype(int)
    df['hour']    = df['time'].apply(lambda x: int(x[11:13]))
    df['minute']  = df['time'].apply(lambda x: int(x[14:15]+'0'))
    df['hourCut'] = df['hour'].map(get_hour_cut)
    
    # count,sum
    # group相当于在minute(10)上统计
    result = df.groupby(['stationID', 'week', 'weekend', 'day', 'hour', 'minute']).status.agg(['count', 'sum']).reset_index()

    #每10mins统计    
    # nunique
    tmp     = df.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result  = result.merge(tmp, on=['stationID'], how='left')
    tmp     = df.groupby(['stationID','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result  = result.merge(tmp, on=['stationID','hour'], how='left')
    tmp     = df.groupby(['stationID','hour','minute'])['deviceID'].nunique().\
                                           reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result  = result.merge(tmp, on=['stationID','hour','minute'], how='left')
    
    # in,out
    result['inNums']  = result['sum']
    result['outNums'] = result['count'] - result['sum']
    
    #
    result['day_since_first'] = result['day'] - 1 
    result.fillna(0, inplace=True)
    del result['sum'],result['count']
    
    return result
data = get_base_features(test_28)

data_list = os.listdir(path+'/Metro_train/')
for i in range(0, len(data_list)):
    if data_list[i].split('.')[-1] == 'csv':
        print(data_list[i], i)
        df = read_data(path+'/Metro_train/' + data_list[i])
        df = get_base_features(df)
        data = pd.concat([data, df], axis=0, ignore_index=True)
    else:
        continue

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
data.loc[data['day']==21,['inNums','outNums']] = 0
data = data[data['day'] != 1]

# 将test拼接到训练数据中
# 有一些列无法产生，大概用了nan
test['week']    = pd.to_datetime(test['startTime']).dt.dayofweek + 1
test['weekend'] = (pd.to_datetime(test.startTime).dt.weekday >=5).astype(int)
test['day']     = test['startTime'].apply(lambda x: int(x[8:10]))
test['hour']    = test['startTime'].apply(lambda x: int(x[11:13]))
test['minute']  = test['startTime'].apply(lambda x: int(x[14:15]+'0'))
test['hourCut'] = test['hour'].map(get_hour_cut)
test['day_since_first'] = test['day'] - 1
test = test.drop(['startTime','endTime'], axis=1)
data = pd.concat([data,test], axis=0, ignore_index=True)
###
## 对stationID标记 删除 stationID=54
data = data[data['stationID']!=54]
##
stat_columns = ['inNums','outNums']

def get_refer_day(d):
    if d == 20:
        return 29
    else:
        return d + 1
# 1->1 1->2 2->3...20->29
tmp = data.copy()
tmp_df = tmp[tmp.day==2]
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

tmp = data.groupby(['stationID','week','hour','minute'], as_index=False)['inNums'].agg({
                                                                        'inNums_whm_max'    : 'max',
                                                                        'inNums_whm_min'    : 'min',
                                                                        'inN    ums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data.groupby(['stationID','week','hour','minute'], as_index=False)['outNums'].agg({
                                                                        'outNums_whm_max'    : 'max',
                                                                        'outNums_whm_min'    : 'min',
                                                                        'outNums_whm_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour','minute'], how='left')

tmp = data.groupby(['stationID','week','hour'], as_index=False)['inNums'].agg({
                                                                        'inNums_wh_max'    : 'max',
                                                                        'inNums_wh_min'    : 'min',
                                                                        'inNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour'], how='left')

tmp = data.groupby(['stationID','week','hour'], as_index=False)['outNums'].agg({
                                                                        #'outNums_wh_max'    : 'max',
                                                                        #'outNums_wh_min'    : 'min',
                                                                        'outNums_wh_mean'   : 'mean'
                                                                        })
data = data.merge(tmp, on=['stationID','week','hour'], how='left')

def recover_day(d):
    if d in [2,3,4]:
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
                num_boost_round=10000,
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
test['outNums'] = gbm.predict(X_test)

sub = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')]
sub['hour']    = sub['startTime'].apply(lambda x: int(x[11:13]))
sub['minute']  = sub['startTime'].apply(lambda x: int(x[14:15]+'0'))
sub['hour_minutes'] = sub['hour']*60 + sub['minute']
sub = sub.drop(columns=['inNums','outNums'])
sub = sub.merge(test,on=['stationID','hour_minutes'],how='left')
sub.loc[sub['stationID']==54, ['inNums','outNums']] = 0
## 对夜晚数据单独处理
data = pd.read_csv('../input/after_base_features.csv')
data['hour_minutes'] = data['hour']*60+data['minute']
data_in = data.loc[(data['hour_minutes']>=1420) | (data['hour_minutes']<=320), ['stationID','day','hour_minutes','inNums','outNums']]
data_out = data.loc[data['hour_minutes']<=350, ['stationID','day','hour_minutes','inNums','outNums']]
data_in = data_in[data_in['inNums'] != 0]
data_out = data_out[data_out['outNums'] != 0]

def special_time(df_data, df_label, inOrout):
    # count
    tmp = df_data.groupby(['stationID', 'hour_minutes'])['day'].count().reset_index(name=f'count_inNum_days_{inOrout}')
    df_label = df_label.merge(tmp, on=['stationID', 'hour_minutes'], how='left')
    #submit.fillnan(0)
    tmp = df_data.groupby(['stationID', 'hour_minutes'])['day'].count().reset_index(name=f'count_outNum_days_{inOrout}')
    df_label = df_label.merge(tmp, on=['stationID', 'hour_minutes'], how='left')
    # mean
    tmp = df_data.groupby(['stationID', 'hour_minutes'])['inNums'].mean().reset_index(name=f'mean_inNums_{inOrout}')
    df_label = df_label.merge(tmp, on=['stationID', 'hour_minutes'], how='left')
    tmp = df_data.groupby(['stationID', 'hour_minutes'])['outNums'].mean().reset_index(name=f'mean_outNums_{inOrout}')
    df_label = df_label.merge(tmp, on=['stationID', 'hour_minutes'], how='left')
    # mode

    return df_label


submit = sub


submit_in = special_time(data_in, submit[(submit['hour_minutes']>=1420) | (submit['hour_minutes']<=320)], 'in')
submit_out = special_time(data_out, submit[submit['hour_minutes']<=350], 'out')

submit_in = submit_in.fillna(0)
submit_out = submit_out.fillna(0)
from datetime import datetime
submit = submit.merge(submit_in, on=['stationID','hour_minutes','startTime','endTime','inNums','outNums'], how='left')
submit = submit.merge(submit_out, on=['stationID','hour_minutes','startTime','endTime','inNums','outNums'], how='left')
submit.loc[(submit['hour_minutes']>=1420) | (submit['hour_minutes']<=320), 'inNums'] = submit.loc[(submit['hour_minutes']>=1420)| (submit['hour_minutes']<=320), 'mean_inNums_in']
submit.loc[submit['hour_minutes']<=350, 'outNums'] = submit.loc[submit['hour_minutes']<=350, 'mean_outNums_out']

submit[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(f'../submit/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)


#sub['inNums']   = test['inNums'].values
#sub['outNums']  = test['outNums'].values
## 结果修正
#sub.loc[sub.inNums<0 , 'inNums']  = 0
#sub.loc[sub.outNums<0, 'outNums'] = 0
#sub[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv('../submit/baseline_fromyu.csv', index=False)