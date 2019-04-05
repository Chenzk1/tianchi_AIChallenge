#%%
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

BASE_PATH = os.path.join('../input')
RAW_PATH = os.path.join(BASE_PATH, 'RAW_DATA')
TRAIN_PATH = os.path.join(RAW_PATH, 'Metro_train')
TEST_A_PATH = os.path.join(RAW_PATH, 'Metro_testA')
SUBMIT_PATH = os.path.join('../submit')
################################
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

def date_processing(data):
    data['startTime'] = data['time'].apply(lambda x: str(x)[:15]+ '0:00')
    data['day'] = data['startTime'].apply(lambda x: int(str(x)[8:10]))
    data['hour'] = data['startTime'].apply(lambda x: int(str(x)[11:13]))
    data['minute'] = data['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
    data['hour_minute'] = data['hour']*60+data['minute']
    data['startTime'] = pd.to_datetime(data['startTime'],format= '%Y-%m-%d %H:%M:%S')
    data['weekday'] = data['startTime'].dt.weekday
    data['hourCut'] = data['hour'].map(get_hour_cut)
    return data

def base_features(data):        
    result = data.groupby(['stationID', 'startTime','day', 'hour', 'minute','weekday'])['status'].agg(['count','sum'])
    result = result.reset_index()
    result['inNums'] = result['sum']
    result['outNums'] = result['count'] - result['sum']
    result = result.drop(columns=['count', 'sum'])
    # id特征
    tmp     = data.groupby(['stationID'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID')
    result  = result.merge(tmp, on=['stationID'], how='left')
    tmp     = data.groupby(['stationID','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour')
    result  = result.merge(tmp, on=['stationID','hour'], how='left')
    tmp     = data.groupby(['stationID','hour','minute'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_hour_minute')
    result  = result.merge(tmp, on=['stationID','hour','minute'], how='left')
    tmp     = data.groupby(['stationID','weekday','hour'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_wh')
    result  = result.merge(tmp, on=['stationID','weekday','hour'], how='left')
    tmp     = data.groupby(['stationID','weekday','hour','minute'])['deviceID'].nunique().reset_index(name='nuni_deviceID_of_stationID_whm')
    result  = result.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')
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
    
    return result

# 数据预处理 stationID!=54 & evening
def preprocess(data):
    data = data[data['stationID']!=54]
    data = data[data['hour_minute'] >= 320]
    return data

## dateprocessing
data = pd.DataFrame()
for file in os.listdir(TRAIN_PATH):
    if file[15:17] in ['01', '05', '06', '12', '13', '19', '20']:
        continue
    print(f'the file: {file}')
    temp = read_data(os.path.join(TRAIN_PATH, file))
    temp = date_processing(temp)
    data = pd.concat([data, temp],ignore_index=True)
    del temp

test_name = os.path.join(TEST_A_PATH, 'testA_record_2019-01-28.csv')
test_28 = read_data(test_name)
test_28 = date_processing(test_28)
data = pd.concat([data, test_28],ignore_index=True)
##process -> base features
data = preprocess(data)
data = base_features(data)
## concat test data
test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
test = pd.read_csv(test_name)
test['time'] = test['startTime']
test = test.drop(columns=['startTime','endTime','inNums', 'outNums'])
test = date_processing(test)

test = preprocess(test)
# 将test与data拼接
temp = data.groupby(['stationID'])['nuni_deviceID_of_stationID'].mean().reset_index()
test = test.merge(temp,on=['stationID'],how='left')
temp = data.groupby(['stationID','hour'])['nuni_deviceID_of_stationID_hour'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour'],how='left')
temp = data.groupby(['stationID','hour','minute'])['nuni_deviceID_of_stationID_hour_minute'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour','minute'],how='left')
temp = data.groupby(['stationID','weekday','hour'])['nuni_deviceID_of_stationID_wh'].mean().reset_index()
test = test.merge(temp,on=['stationID','weekday','hour'],how='left')
temp = data.groupby(['stationID','weekday','hour','minute'])['nuni_deviceID_of_stationID_whm'].mean().reset_index()
test = test.merge(temp,on=['stationID','weekday','hour','minute'],how='left')

temp = data.groupby(['stationID'])['most_deviceID_of_stationID'].mean().reset_index()
test = test.merge(temp,on=['stationID'],how='left')
temp = data.groupby(['stationID','hour'])['most_deviceID_of_stationID&hour'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour'],how='left')
temp = data.groupby(['stationID','hour','minute'])['most_deviceID_of_stationID&wh'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour','minute'],how='left')
temp = data.groupby(['stationID'])['most_payType_of_stationID'].mean().reset_index()
test = test.merge(temp,on=['stationID'],how='left')
temp = data.groupby(['stationID','hour'])['most_payType_of_stationID&hour'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour'],how='left')
temp = data.groupby(['stationID','hour','weekday'])['most_payType_of_stationID&wh'].mean().reset_index()
test = test.merge(temp,on=['stationID','hour','weekday'],how='left')

test = test.drop(columns='time')
data['hour_minute'] = data['hour']*60+data['minute']
data['hourCut'] = data['hour'].map(get_hour_cut)


data = pd.concat([data, test],ignore_index=True)
data = data.fillna(0)
#######################################################111
def baseline_yu(data):
    # 剔除周末,修改为连续时间
    def fix_day(d):
        if d in [1,2,3,4]:
            return d
        elif d in [7,8,9,10,11]:
            return d - 2
        elif d in [14,15,16,17,18]:
            return d - 4
        elif d in [21,22,23,24,25]:
            return d - 6
        elif d in [28,29]:
            return d - 8
    data['day'] = data['day'].apply(fix_day)
    data.loc[data['day']==21,['inNums','outNums']] = 0

    stat_columns = ['inNums','outNums']

    def get_refer_day(d):
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
    data['count_last'] = data['inNums_last'] + data['outNums_last']
    data.fillna(0, inplace=True)
    ####### 构造统计特征
    data['count'] = data['inNums'] + data['outNums']
    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour','minute'], as_index=False)['count'].agg({
                                                                            'count_whm_max'    : 'max',
                                                                            'count_whm_min'    : 'min',
                                                                            'count_whm_mean'   : 'mean'
                                                                            })
    train = data.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour'], as_index=False)['count'].agg({
                                                                            'count_wh_max'    : 'max',
                                                                            'count_wh_min'    : 'min',
                                                                            'count_wh_mean'   : 'mean'
                                                                            })
    train = train.merge(tmp, on=['stationID','weekday','hour'], how='left')
    tmp = data[data['day']!=21].groupby(['stationID','hour'], as_index=False)['count'].agg({
                                                                            'count_h_max'    : 'max',
                                                                            'count_h_min'    : 'min',
                                                                            'count_h_mean'   : 'mean'
                                                                            })
    train = train.merge(tmp, on=['stationID','hour'], how='left')
    ##
    
    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour','minute'], as_index=False)['inNums'].agg({
                                                                            'inNums_whm_max'    : 'max',
                                                                            'inNums_whm_min'    : 'min',
                                                                            'inNums_whm_mean'   : 'mean'
                                                                            })
    train = data.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour','minute'], as_index=False)['outNums'].agg({
                                                                            'outNums_whm_max'    : 'max',
                                                                            'outNums_whm_min'    : 'min',
                                                                            'outNums_whm_mean'   : 'mean'
                                                                            })
    train = train.merge(tmp, on=['stationID','weekday','hour','minute'], how='left')

    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour'], as_index=False)['inNums'].agg({
                                                                            'inNums_wh_max'    : 'max',
                                                                            'inNums_wh_min'    : 'min',
                                                                            'inNums_wh_mean'   : 'mean'
                                                                            })
    train = train.merge(tmp, on=['stationID','weekday','hour'], how='left')

    tmp = data[data['day']!=21].groupby(['stationID','weekday','hour'], as_index=False)['outNums'].agg({
                                                                            #'outNums_wh_max'    : 'max',
                                                                            #'outNums_wh_min'    : 'min',
                                                                            'outNums_wh_mean'   : 'mean'
                                                                            })
    train = train.merge(tmp, on=['stationID','weekday','hour'], how='left')
    ####### 对stationID标记
    # outNums
    max_station_out = train.groupby(['stationID', 'day'])['outNums'].max().reset_index(name='max_stationID_outNums')
    mean_station = max_station_out.groupby(['stationID'])['max_stationID_outNums'].mean().reset_index(name='mean_stationID_outNums')
    flag_1 = mean_station[(mean_station['mean_stationID_outNums']<=150)].index.tolist()
    flag_2 = mean_station[(mean_station['mean_stationID_outNums']>150) & (mean_station['mean_stationID_outNums']<=500)].index.tolist()
    flag_3 = mean_station[(mean_station['mean_stationID_outNums']>500) & (mean_station['mean_stationID_outNums']<=1100)].index.tolist()
    flag_4 = mean_station[(mean_station['mean_stationID_outNums']>1100)].index.tolist()
    train['stationID_falg_out'] = 0
    i=1
    for flag in [flag_2, flag_3, flag_4]:
        for s_ID in flag:
            train.loc[train['stationID']==s_ID, 'stationID_falg_out'] = i
        i += 1
    train.loc[train['stationID']==9, 'stationID_falg_out'] = 4
    train.loc[train['stationID']==15, 'stationID_falg_out'] = 5
    # inNums
    max_station_in= train.groupby(['stationID', 'day'])['inNums'].max().reset_index(name='max_stationID_inNums')
    mean_station = max_station_in.groupby(['stationID'])['max_stationID_inNums'].mean().reset_index(name='mean_stationID_inNums')
    flag_1 = mean_station[(mean_station['mean_stationID_inNums']<=150)].index.tolist()
    flag_2 = mean_station[(mean_station['mean_stationID_inNums']>150) & (mean_station['mean_stationID_inNums']<=500)].index.tolist()
    flag_3 = mean_station[(mean_station['mean_stationID_inNums']>500) & (mean_station['mean_stationID_inNums']<=1000)].index.tolist()
    flag_4 = mean_station[(mean_station['mean_stationID_inNums']>1000)].index.tolist()
    train['stationID_falg_in'] = 0
    i=1
    for flag in [flag_2, flag_3, flag_4]:
        for s_ID in flag:
            train.loc[train['stationID']==s_ID, 'stationID_falg_in'] = i
        i += 1
    train.loc[train['stationID']==9, 'stationID_falg_in'] = 4
    train.loc[train['stationID']==15, 'stationID_falg_in'] = 5
    
    # recover_day
    def recover_day(d):
        if d in [2,3,4]:
            return d
        elif d in [5,6,7,8,9]:
            return d + 2
        elif d in [10,11,12,13,14]:
            return d + 4
        elif d in [15,16,17,18,19]:
            return d + 6
        elif d in [20,21]:
            return d + 8
        else:
            return d
    train['day'] = train['day'].apply(recover_day)
    #############################################
    removed_cols = ['startTime','weekend','inNums','outNums','count',
                    'stationID','most_payType_of_stationID','most_payType_of_stationID&hour','most_payType_of_stationID&wh',
                    ]
    all_columns = [f for f in train.columns if f not in removed_cols]
    all_data = train[train.day!=29]
    X_data = all_data[all_columns]
    y_data = all_data[['inNums','outNums']]

    train_data = train[(train.day<28)]
    X_train = train_data[all_columns]
    y_train = train_data[['inNums','outNums']]

    valid_data = train[train.day==28]
    X_valid = valid_data[all_columns]
    y_valid = valid_data[['inNums','outNums']]

    test_data  = train[train.day==29]# test_data出问题了
    X_test = test_data[all_columns]
    y_test = test_data[['inNums','outNums']]

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
    y_train_in = y_train['inNums']
    y_valid_in = y_valid['inNums']
    y_data_in  = y_data['inNums']
    lgb_train = lgb.Dataset(X_train, y_train_in)
    lgb_evals = lgb.Dataset(X_valid, y_valid_in , reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    importance_in = pd.DataFrame()
    importance_in['feature'] = gbm.feature_name()
    importance_in['importance'] = gbm.feature_importance()/2
    ### all_data
    lgb_train = lgb.Dataset(X_data, y_data_in)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=1000,
                    )
    test_data['inNums_pre'] = gbm.predict(X_test)
    valid_data['inNums_pre'] = gbm.predict(X_valid)
    importance_in['importance'] += gbm.feature_importance()/2
    ######################################################outNums
    y_train_out = y_train['outNums']
    y_valid_out = y_valid['outNums']
    y_data_out  = y_data['outNums']
    lgb_train = lgb.Dataset(X_train, y_train_out)
    lgb_evals = lgb.Dataset(X_valid, y_valid_out , reference=lgb_train)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=200,
                    verbose_eval=1000,
                    )
    importance_out = pd.DataFrame()
    importance_out['feature'] = gbm.feature_name()
    importance_out['importance'] = gbm.feature_importance()/2
    ### all_data
    lgb_train = lgb.Dataset(X_data, y_data_out)
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=gbm.best_iteration,
                    valid_sets=[lgb_train],
                    valid_names=['train'],
                    verbose_eval=1000,
                    )
    test_data['outNums_pre'] = gbm.predict(X_test)
    valid_data['outNums_pre'] = gbm.predict(X_valid)
    importance_out['importance'] += gbm.feature_importance()/2
    ################################生成提交文件
    def submit_file(test_data):
        ''''inNums_pre','outNums_pre',完成夜晚数据和stationID=54拼接'''
        test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
        submit = pd.read_csv(test_name)
        submit['hour'] = submit['startTime'].apply(lambda x: int(str(x)[11:13]))
        submit['minute'] = submit['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
        submit['hour_minute'] = submit['hour']*60 + submit['minute']
        
        submit = submit.merge(test_data[['stationID','hour','minute','inNums_pre','outNums_pre']], on=['stationID','hour','minute'], how='left')
        submit.inNums = submit.inNums_pre
        submit.outNums = submit.outNums_pre
        submit.drop(columns=['inNums_pre','outNums_pre'],inplace=True)
        ####后续：拼接夜晚数据和stationID=54
        ###夜晚
        submit_ = pd.read_csv('../submit/submit_28_change.csv')
        submit_['hour'] = submit_['startTime'].apply(lambda x: int(str(x)[11:13]))
        submit_['minute'] = submit_['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
        submit_['hour_minute'] = submit_['hour']*60 + submit_['minute']
        submit.loc[(submit['hour_minute']>=1420) | (submit['hour_minute']<=320), 'inNums'] = submit_.loc[(submit_['hour_minute']>=1420)| (submit_['hour_minute']<=320), 'inNums']
        submit.loc[submit['hour_minute']<=350, 'outNums'] = submit_.loc[submit['hour_minute']<=350, 'outNums']
        #####54
        submit = submit.fillna(0)
        submit['inNums'] = submit['inNums'].apply(lambda x:np.clip(x,0,4000))
        submit['outNums'] = submit['outNums'].apply(lambda x:np.clip(x,0,4000))
        submit[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(f'../submit/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)


plt.plot(submit_.loc[submit_['stationID']==1,'hour_minute'],submit_.loc[submit_['stationID']==1,'inNums'])