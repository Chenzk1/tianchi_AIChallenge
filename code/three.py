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
data = pd.read_csv('../input/111.csv')
#######################################################111
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
######################## create features
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
def create_features(df_label, df_train,slip):
    # nums
    #这里加入前一天的数据
    if slip==1:
        size = [1,5,7]
    elif slip==2:
        size = [2,6]
    elif slip==3:
        size = [3,9]
    elif slip==5:
        size = [5,8]
    for i in size:
        if df_train.day_gap.min() > -i:
            break
        df_select=df_train[df_train.day_gap>=-i].copy()
        if i==1:
            df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"inNums", "inNums_mean_s_h_m_%s"%i)
            df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"outNums", "outNums_mean_s_h_m_%s"%i)
            continue        
       
        # stationID weekday hour
        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_mean_s_w_h_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_median_s_w_h_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour'],"inNums", "inNums_max_s_w_h_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_mean_s_w_h_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_median_s_w_h_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour'],"outNums", "outNums_max_s_w_h_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_mean_s_w_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_median_s_w_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_max_s_w_hm_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour','minute'],"inNums", "inNums_min_s_w_hm_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_mean_s_w_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_median_s_w_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_max_s_w_hm_%s"%i)
        df_label=feat_min(df_label,df_select,["stationID", 'weekday','hour','minute'],"outNums", "outNums_min_s_w_hm_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"inNums", "inNums_mean_s_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'hour','minute'],"inNums", "inNums_median_s_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'hour','minute'],"inNums", "inNums_max_s_hm_%s"%i)

        df_label=feat_mean(df_label,df_select,["stationID", 'hour','minute'],"outNums", "outNums_mean_s_hm_%s"%i)
        df_label=feat_median(df_label,df_select,["stationID", 'hour','minute'],"outNums", "outNums_median_s_hm_%s"%i)
        df_label=feat_max(df_label,df_select,["stationID", 'hour','minute'],"outNums", "outNums_max_s_hm_%s"%i)

        # hourcut 在df_label中加入hour_cut
        # 标记stationID    
    df_label['hour_minute_cut'] = pd.cut(df_label['hour_minute'],5,labels=False)

        ####### 对stationID标记
    # outNums
    max_station_out = df_label.groupby(['stationID', 'day'])['outNums'].mean().reset_index(name='max_stationID_outNums')
    mean_station = max_station_out.groupby(['stationID'])['max_stationID_outNums'].mean().reset_index(name='mean_stationID_outNums')
    flag_1 = mean_station[(mean_station['mean_stationID_outNums']<=150)].index.tolist()
    flag_2 = mean_station[(mean_station['mean_stationID_outNums']>150) & (mean_station['mean_stationID_outNums']<=500)].index.tolist()
    flag_3 = mean_station[(mean_station['mean_stationID_outNums']>500) & (mean_station['mean_stationID_outNums']<=1100)].index.tolist()
    flag_4 = mean_station[(mean_station['mean_stationID_outNums']>1100)].index.tolist()
    df_label['stationID_falg_out'] = 0
    i=1
    for flag in [flag_2, flag_3, flag_4]:
        for s_ID in flag:
            df_label.loc[df_label['stationID']==s_ID, 'stationID_falg_out'] = i
        i += 1
    df_label.loc[df_label['stationID']==9, 'stationID_falg_out'] = 4
    df_label.loc[df_label['stationID']==15, 'stationID_falg_out'] = 5
    # inNums
    max_station_in= df_label.groupby(['stationID', 'day'])['inNums'].max().reset_index(name='max_stationID_inNums')
    mean_station = max_station_in.groupby(['stationID'])['max_stationID_inNums'].mean().reset_index(name='mean_stationID_inNums')
    flag_1 = mean_station[(mean_station['mean_stationID_inNums']<=150)].index.tolist()
    flag_2 = mean_station[(mean_station['mean_stationID_inNums']>150) & (mean_station['mean_stationID_inNums']<=500)].index.tolist()
    flag_3 = mean_station[(mean_station['mean_stationID_inNums']>500) & (mean_station['mean_stationID_inNums']<=1000)].index.tolist()
    flag_4 = mean_station[(mean_station['mean_stationID_inNums']>1000)].index.tolist()
    df_label['stationID_falg_in'] = 0
    i=1
    for flag in [flag_2, flag_3, flag_4]:
        for s_ID in flag:
            df_label.loc[df_label['stationID']==s_ID, 'stationID_falg_in'] = i
        i += 1
    df_label.loc[df_label['stationID']==9, 'stationID_falg_in'] = 4
    df_label.loc[df_label['stationID']==15, 'stationID_falg_in'] = 5
    
    return df_label
########################
# 20作验证 21测试
for slip in [1]:
    if slip==1:
        i_end = 10
    elif slip==2:
        i_end = 5
    elif slip==3:
        i_end = 3
    elif slip==5:
        i_end = 2
        
    df_train = data
    t_end = 20
    nday = slip
    all_data = []
    i = 0
    # 2~19 18天的数据
    removed_cols = ['weekend','most_payType_of_stationID',
                    'most_payType_of_stationID&hour','most_payType_of_stationID&wh',]
    all_columns = [f for f in df_train.columns if f not in removed_cols]
    for day_gap in range(nday*1, nday*(18//nday+1), nday):
        if i>=i_end:
            break
        i+=1
        t_begin = t_end - day_gap
        print(t_begin)
        df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
        df_feature=df_train[df_train.day_gap<0].copy()
        df_label=df_train[(df_train.day_gap>=0)&(df_train.day_gap<nday)][all_columns].copy()
        train_data_tmp = create_features(df_label,df_feature,slip)
        all_data.append(train_data_tmp)
    train=pd.concat(all_data)
    # 2~20 19天的数据
    del all_data
    all_data = []
    i=0
    for day_gap in range(nday*1, nday*(19//nday+1), nday):
        if i>=i_end:
            break
        i+=1
        t_begin = 21 - day_gap
        print(t_begin)
        df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
        df_feature=df_train[df_train.day_gap<0].copy()
        df_label=df_train[(df_train.day_gap>=0)&(df_train.day_gap<nday)][all_columns].copy()
        train_data_tmp=create_features(df_label,df_feature,slip)
        all_data.append(train_data_tmp)
    train_all=pd.concat(all_data)
    #构造线下验证集
    t_begin=20
    print(f'valid: {t_begin}')
    df_label=df_train[df_train['day']==20]
    df_label["day_gap"]=0
    df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
    df_label=df_label[all_columns].copy()
    valid=create_features(df_label,df_train,slip)
    #构造线上测试集
    t_begin=21
    print(f'test: {t_begin}')
    df_label=df_train[df_train['day']==21]
    df_label["day_gap"]=0
    df_train["day_gap"]=df_train["day"].apply(lambda x:int(x-t_begin))
    df_label=df_label[all_columns].copy()
    test=create_features(df_label,df_train,slip)


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
    train_all['day'] = train_all['day'].apply(recover_day)    
    train['day'] = train['day'].apply(recover_day)
    valid['day'] = valid['day'].apply(recover_day)
    test['day'] = test['day'].apply(recover_day)
    #############################################
    removed_cols = ['startTime','weekend','inNums','outNums',
                    'stationID','most_payType_of_stationID','most_payType_of_stationID&hour','most_payType_of_stationID&wh',
                    ]
    all_columns = [f for f in train.columns if f not in removed_cols]
    print(f'all_train_data:{train_all.day.unique()}')
    X_data = train_all[all_columns]
    y_data = train_all[['inNums','outNums']]
    print(f'train_data:{train.day.unique()}')
    train_data = train
    X_train = train_data[all_columns]
    y_train = train_data[['inNums','outNums']]
    print(f'valid_data:{valid.day.unique()}')
    valid_data = valid
    X_valid = valid_data[all_columns]
    y_valid = valid_data[['inNums','outNums']]
    print(f'test_data:{test.day.unique()}')
    test_data = test
    X_test = test_data[all_columns]
    y_test = test_data[['inNums','outNums']]

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_seed':0,
        'bagging_freq': 2,
        'verbose': 500,
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
                    num_boost_round=100000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=500,
                    verbose_eval=600,
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
                    verbose_eval=500,
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
                    num_boost_round=100000,
                    valid_sets=[lgb_train,lgb_evals],
                    valid_names=['train','valid'],
                    early_stopping_rounds=500,
                    verbose_eval=600,
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
                    verbose_eval=500,
                    )
    test_data['outNums_pre'] = gbm.predict(X_test)
    valid_data['outNums_pre'] = gbm.predict(X_valid)
    importance_out['importance'] += gbm.feature_importance()/2
    
    #test_data[['stationID', 'startTime', 'inNums_pre', 'outNums_pre']].to_csv(f'../submit/330/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)



#sub = pd.DataFrame()
#i=0
#for file in os.listdir('../submit/330/'):
#    print(f'the file: {file}')
#    temp = pd.read_csv('../submit/330/'+file)
#    if i==0:
#        i = 1
#        sub = temp 
#    else:
#        sub['inNums_pre'] += temp['inNums_pre']
#        sub['outNums_pre'] += temp['outNums_pre']
#files = [file for file in os.listdir('../submit') if file[-4:]=='.csv']
#len_files = len(files)
#sub['inNums_pre'] = sub['inNums_pre'].apply(lambda x: x/len_files)
#sub['outNums_pre'] = sub['outNums_pre'].apply(lambda x: x/len_files)
#sub['hour'] = sub['startTime'].apply(lambda x: int(str(x)[11:13]))
#sub['minute'] = sub['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
#sub['hour_minute'] = sub['hour']*60 + sub['minute']

#def submit_file(test_data):
#    ''''inNums_pre','outNums_pre',完成夜晚数据和stationID=54拼接'''
#    test_name = os.path.join(TEST_A_PATH, 'testA_submit_2019-01-29.csv')
#    submit = pd.read_csv(test_name)
#    submit['hour'] = submit['startTime'].apply(lambda x: int(str(x)[11:13]))
#    submit['minute'] = submit['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
#    submit['hour_minute'] = submit['hour']*60 + submit['minute']
#    
#    submit = submit.merge(test_data[['stationID','hour','minute','inNums_pre','outNums_pre']], on=['stationID','hour','minute'], how='left')
#    submit.inNums = submit.inNums_pre
#    submit.outNums = submit.outNums_pre
#    submit.drop(columns=['inNums_pre','outNums_pre'],inplace=True)
#    ####后续：拼接夜晚数据和stationID=54
#    ###夜晚
#    submit_ = pd.read_csv('../submit/submit_28_change.csv')
#    submit_['hour'] = submit_['startTime'].apply(lambda x: int(str(x)[11:13]))
#    submit_['minute'] = submit_['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
#    submit_['hour_minute'] = submit_['hour']*60 + submit_['minute']
#    submit.loc[(submit['hour_minute']>=1420) | (submit['hour_minute']<=320), 'inNums'] = submit_.loc[(submit_['hour_minute']>=1420)| (submit_['hour_minute']<=320), 'inNums']
#    submit.loc[submit['hour_minute']<=350, 'outNums'] = submit_.loc[submit['hour_minute']<=350, 'outNums']
#    #####54
#    submit = submit.fillna(0)
#    submit['inNums'] = submit['inNums'].apply(lambda x:np.clip(x,0,4000))
#    submit['outNums'] = submit['outNums'].apply(lambda x:np.clip(x,0,4000))
#    submit[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(f'../submit/submit_{datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")}.csv', index=False)
#
#submit_file(sub)
#
#submit = pd.read_csv('../submit/submit_2019_03_30_11_54.csv')
#submit['day'] = submit['startTime'].apply(lambda x: int(str(x)[8:10]))
#submit['hour'] = submit['startTime'].apply(lambda x: int(str(x)[11:13]))
#submit['minute'] = submit['startTime'].apply(lambda x: int(str(x)[14:15]+'0'))# hour+10min 10min最后可以删除
#submit['hour_minute'] = submit['hour']*60 + submit['minute']
#
#plt.plot(submit.loc[(submit['day']==29) & (submit['stationID']==15), 'hour_minute'], submit.loc[(submit['day']==29) & (submit['stationID']==15), 'outNums'])
