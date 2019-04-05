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
    data['hourCut'] = data['hour'].map(get_hour_cut)

    return data

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
test['time'] = test['startTime']
test = test.drop(columns=['startTime','endTime','inNums', 'outNums'])
test = date_processing(test)
data = pd.concat([data, test],ignore_index=True)
data = data.fillna(0)

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

data = base_features(data)
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
    elif d in [28,29]:
        return d - 8
data['day'] = data['day'].apply(fix_day)
data.loc[data['day']==21,['inNums','outNums']] = 0
data = data[data['day'] != 1]
## 对stationID标记 删除 stationID=54
data = data[data['stationID']!=54]
# outNums
max_station_out = data.groupby(['stationID', 'day'])['outNums'].max().reset_index(name='max_stationID_outNums')
mean_station = max_station_out.groupby(['stationID'])['max_stationID_outNums'].mean().reset_index(name='mean_stationID_outNums')
flag_1 = mean_station[(mean_station['mean_stationID_outNums']<=150)].index.tolist()
flag_2 = mean_station[(mean_station['mean_stationID_outNums']>150) & (mean_station['mean_stationID_outNums']<=500)].index.tolist()
flag_3 = mean_station[(mean_station['mean_stationID_outNums']>500) & (mean_station['mean_stationID_outNums']<=1100)].index.tolist()
flag_4 = mean_station[(mean_station['mean_stationID_outNums']>1100)].index.tolist()
data['stationID_falg_out'] = 0
i=1
for flag in [flag_2, flag_3, flag_4]:
    for s_ID in flag:
        data.loc[data['stationID']==s_ID, 'stationID_falg_out'] = i
    i += 1
data.loc[data['stationID']==9, 'stationID_falg_out'] = 4
data.loc[data['stationID']==15, 'stationID_falg_out'] = 5
# inNums
max_station_in= data.groupby(['stationID', 'day'])['inNums'].max().reset_index(name='max_stationID_inNums')
mean_station = max_station_in.groupby(['stationID'])['max_stationID_inNums'].mean().reset_index(name='mean_stationID_inNums')
flag_1 = mean_station[(mean_station['mean_stationID_inNums']<=150)].index.tolist()
flag_2 = mean_station[(mean_station['mean_stationID_inNums']>150) & (mean_station['mean_stationID_inNums']<=500)].index.tolist()
flag_3 = mean_station[(mean_station['mean_stationID_inNums']>500) & (mean_station['mean_stationID_inNums']<=1000)].index.tolist()
flag_4 = mean_station[(mean_station['mean_stationID_inNums']>1000)].index.tolist()
data['stationID_falg_in'] = 0
i=1
for flag in [flag_2, flag_3, flag_4]:
    for s_ID in flag:
        data.loc[data['stationID']==s_ID, 'stationID_falg_in'] = i
    i += 1
data.loc[data['stationID']==9, 'stationID_falg_in'] = 4
data.loc[data['stationID']==15, 'stationID_falg_in'] = 5
## create features
def feat_count(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].count()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_count" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_nunique(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].nunique()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_nunique" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_mean(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].mean()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_mean" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_std(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].std()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_std" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_median(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].median()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_median" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_max(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].max()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_max" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_min(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].min()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_min" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_sum(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].sum()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_sum" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_var(df, df_feature, fe,value,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].var()).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_var" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df

def feat_quantile(df, df_feature, fe,value,n,name=""):
    df_count = pd.DataFrame(df_feature.groupby(fe)[value].quantile(n)).reset_index()
    if not name:
        df_count.columns = fe + [value+"_%s_quantile" % ("_".join(fe))]
    else:
        df_count.columns = fe + [name]
    df = df.merge(df_count, on=fe, how="left").fillna(0)
    return df
#['stationID', 'startTime', 'day', 'hour', 'minute', 'weekday', 'inNums', 'outNums',  'day_gap']
def create_features(df_label, df_train):
    # nums
    #这里加入前一天的数据

    for i in [1,3,5,10,15]:
        if df_train.day_gap.min() > -i:
            break
        df_select=df_train[df_train.day_gap>=-i].copy()
        if i==1:
            df_label = feat_mean(df_label,df_select,["stationID"],"inNums", "inNums_mean_stationID_%s"%i)
            df_label = feat_mean(df_label,df_select,["stationID"],"outNums", "outNums_mean_stationID_%s"%i)
            df_label=feat_mean(df_label,df_select,["stationID", 'hour'],"inNums", "inNums_mean_s_h_%s"%i)
            df_label=feat_mean(df_label,df_select,["stationID", 'hour'],"outNums", "outNums_mean_s_h_%s"%i)
            df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"inNums", "inNums_mean_s_h_m_%s"%i)
            df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"outNums", "outNums_mean_s_h_m_%s"%i)
            continue        
        # stationID
        df_label=feat_mean(df_label,df_select,["stationID"],"inNums", "inNums_mean_stationID_%s"%i)
        df_label=feat_std(df_label,df_select,["stationID"],"inNums", "inNums_std_stationID_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID"],"inNums", "inNums_median_stationID_%s"%i)
        #df_label=feat_max(df_label,df_select,["stationID"],"inNums", "inNums_max_stationID_%s"%i)
        #df_label=feat_min(df_label,df_select,["stationID"],"inNums", "inNums_min_stationID_%s"%i)
        df_label=feat_var(df_label,df_select,["stationID"],"inNums", "inNums_var_stationID_%s"%i)
        #df_label=feat_quantile(df_label,df_select,["stationID"],"inNums", "inNums_quantile_stationID_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID"],"outNums", "outNums_mean_stationID_%s"%i)
        df_label=feat_std(df_label,df_select,["stationID"],"outNums", "outNums_std_stationID_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID"],"outNums", "outNums_median_stationID_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID"],"outNums", "outNums_max_stationID_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID"],"outNums", "outNums_min_stationID_%s"%i)
        df_label=feat_var(df_label,df_select,["stationID"],"outNums", "outNums_var_stationID_%s"%i)
        #df_label=feat_quantile(df_label,df_select,["stationID"],"outNums", "outNums_quantile_stationID_%s"%i)
       
        # stationID weekday hour
        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_mean_s_w_h_%s"%i)
        df_label=feat_std(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_std_s_w_h_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_median_s_w_h_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_max_s_w_h_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_min_s_w_h_%s"%i)
        df_label=feat_var(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_var_s_w_h_%s"%i)
        #df_label=feat_quantile(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_quantile_s_w_h_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_mean_s_w_h_%s"%i)
        df_label=feat_std(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_std_s_w_h_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_median_s_w_h_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_max_s_w_h_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_min_s_w_h_%s"%i)
        df_label=feat_var(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_var_s_w_h_%s"%i)
        #f_label=feat_quantile(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_quantile_s_w_h_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_mean_s_w_hm_%s"%i)
        #df_label=feat_std(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_std_s_w_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_median_s_w_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_max_s_w_hm_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_min_s_w_hm_%s"%i)
        #df_label=feat_var(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_var_s_w_hm_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_mean_s_w_hm_%s"%i)
        #df_label=feat_std(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_std_s_w_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_median_s_w_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_max_s_w_hm_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_min_s_w_hm_%s"%i)
        #df_label=feat_var(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_var_s_w_hm_%s"%i)
    return df_label


df_train = data
# 根据划窗计算id&num特征
# 按出入车站group，看两站进出时间的统计特征
for slip in [2,3,5,7]:
    print(f'the slip is: {slip}')
    t_end = 21
    nday = slip

    # 构造训练集
    all_data = []
    for i in range(nday*1, nday*(19//nday+1),nday):
        t_begin = t_end-i
        print(t_begin)
        df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
        df_feature=df_train[df_train.day_gap<0].copy()
        df_label=df_train[(df_train.day_gap>=0)&(df_train.day_gap<nday)][["stationID","startTime",'weekday','hour',
        'minute','inNums','outNums']].copy()
        train_data_tmp=create_features(df_label,df_feature)
        all_data.append(train_data_tmp)
    train=pd.concat(all_data)
    #构造线上测试集
    t_begin=21
    print(t_begin)
    df_label=df_train.loc[df_train['day']==21, ['stationID','startTime','weekday','hour','minute','inNums','outNums']]
    df_label["day_gap"]=0
    df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
    df_label=df_label[['stationID','startTime','weekday','hour','minute','inNums','outNums']].copy()
    test=create_features(df_label,df_train)

    #save features data for stacking
    #train.to_csv("../stacking/train.csv",index=None)
    #test.to_csv("../stacking/test.csv",index=None)

    #训练预测
    #weight_df=train[["day_gap"]].copy()
    #weight_df["weight"]=weight_df["day_gap"].apply(lambda x: 1 if x<=6 else 1)
    def stacking(reg, train_data, test_data, reg_name, inOrout, params):
        train_pre = np.zeros(train_data.shape[0])
        test_pre = np.zeros(test_data.shape[0])
        cv_score = []
        
        all_cols = [col for col in train_data.columns if col not in ['inNums', 'outNums']]
        train_x = train_data[all_cols].values
        train_y = train[inOrout].values
        test_data = test_data[all_cols].values
        for i, (trn_index, val_index) in enumerate(kf.split(train_data)):
            trn_x = train_x[trn_index]
            trn_y = train_y[trn_index]
            
            val_x = train_x[val_index]
            val_y = train_y[val_index]
            #weight_train=weight_df.iloc[trn_index]
            #weight_test=weight_df.iloc[val_index]

            trn_matrix = reg.Dataset(trn_x, label=trn_y)
            val_matrix = reg.Dataset(val_x, label=val_y)
            num_round = 200000
            early_stopping_rounds = 500
            if val_matrix:
                model = reg.train(params, trn_matrix, num_round, valid_sets=[trn_matrix, val_matrix],
                                  early_stopping_rounds=early_stopping_rounds, verbose_eval=500
                                  )
                pre= model.predict(val_x,num_iteration=model.best_iteration)
                train_pre[val_index]=pre
                test_pre += (model.predict(test_data, num_iteration=model.best_iteration)) / folds
                cv_score.append(mean_absolute_error(val_y, pre))

            #print(f"folds {i} of {reg_name} score is: {mean_absolute_error(val_y, pre)}")
            
        print("%s_score_list:"%reg_name,cv_score)
        print("%s_score_mean:"%reg_name,np.mean(cv_score))

        return train_pre.reshape(-1,1), test_pre.reshape(-1,1), np.mean(cv_score)

    def lgb_reg(train, test):
        params = {
            'boosting': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 63,
            'learning_rate': 0.08,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_seed':0,
            'bagging_freq': 1,
            'reg_alpha':1,
            'reg_lambda':2,
            'verbose':500,
            'num_threads': 4
        }
        lgb_train_in, lgb_test_in, cv_scores_in = stacking(lgb,train,test,"lgb", 'inNums', params)
        lgb_train_out, lgb_test_out, cv_scores_out = stacking(lgb,train,test,"lgb", 'outNums', params)
        return lgb_train_in, lgb_test_in, cv_scores_in, lgb_train_out, lgb_test_out, cv_scores_out

    import lightgbm as lgb
    folds = 10
    seed = 2019

    #生成数据
    # 考虑不去stationID

    train_data = train.drop(columns=['stationID','startTime'])
    test_data = test.drop(columns=['stationID','startTime'])

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    lgb_train_in, lgb_test_in, cv_scores_in, lgb_train_out, lgb_test_out, cv_scores_out=lgb_reg(train_data,test_data)

    #生成线下
    train["inNums_pre"]=np.clip(lgb_train_in,0,5000)
    train["outNums_pre"]=np.clip(lgb_train_out,0,5000)
    score_result=mean_absolute_error(train["inNums_pre"], train["inNums"]) + mean_absolute_error(train["outNums_pre"], train["outNums"]) 
    print(f'slip {slip}: the total mae score is {score_result/2}')
    #生成提交
    test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
    submit = pd.read_csv(test_name)
    
    #lgb_test_in = np.round(lgb_test_in)
    #lgb_test_out = np.round(lgb_test_out)
    submit['inNums'] = np.clip(lgb_test_in,0,5000)
    submit['outNums'] = np.clip(lgb_test_out,0,5000)
    submit.to_csv(f'../submit/{slip}slips_{folds}folds.csv', index=False)
    
test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
submit = pd.read_csv(test_name)
i = 0
for file in os.listdir('../submit'):
    print(f'the file: {file}')
    temp = pd.read_csv('../submit/'+file)
    if i==0:
        submit['inNums'] = temp['inNums']
        submit['outNums'] = temp['outNums']
        i = 1
    else:
        submit['inNums'] += temp['inNums']
        submit['outNums'] += temp['outNums']
files = [file for file in os.listdir('../submit') if file[-4:]=='.csv']
len_files = len(files)
submit['inNums'] = submit['inNums'].apply(lambda x: x/len_files)
submit['outNums'] = submit['outNums'].apply(lambda x: x/len_files)
submit.to_csv(f'../submit/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)

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


submit = pd.read_csv('../submit/submit_2019_03_28_09_10.csv')
submit['hour'] = submit['startTime'].apply(lambda x: int(str(x)[11:13]))
submit['minute'] = submit['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
submit['hour_minutes'] = submit['hour']*60+submit['minute']

submit_in = special_time(data_in, submit[(submit['hour_minutes']>=1420) | (submit['hour_minutes']<=320)], 'in')
submit_out = special_time(data_out, submit[submit['hour_minutes']<=350], 'out')

submit_in = submit_in.fillna(0)
submit_out = submit_out.fillna(0)

submit = submit.merge(submit_in, on=['stationID','hour_minutes','startTime','endTime','inNums','outNums'], how='left')
submit = submit.merge(submit_out, on=['stationID','hour_minutes','startTime','endTime','inNums','outNums'], how='left')
submit.loc[(submit['hour_minutes']>=1420) | (submit['hour_minutes']<=320), 'inNums'] = submit.loc[(submit['hour_minutes']>=1420)| (submit['hour_minutes']<=320), 'mean_inNums_in']
submit.loc[submit['hour_minutes']<=350, 'outNums'] = submit.loc[submit['hour_minutes']<=350, 'mean_outNums_out']

submit[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(f'../submit/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)


# 对outnums不按stationID groupby
# 对connnect的station求取同用户进出的时间统计量 中值，均值，四分之一，众数etc
# 按照这些统计特征shift时间轴，利用inNums

# 始发站 换乘站 普通站标记（按map or count)

# 特征分析及选择
# 多模型 xgb cgb gbdt 其他回归
# 调参

# 最大值 log