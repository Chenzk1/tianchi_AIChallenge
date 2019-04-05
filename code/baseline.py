# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 9:38 PM
# @Author  : Inf.Turing
# @Site    : 
# @File    : baseline.py
# @Software: PyCharm

import pandas as pd

path = '/Users/inf/PycharmProject/kaggle/dt_flow/data'
test = pd.read_csv(path + '/Metro_testA/testA_submit_2019-01-29.csv')
test_a_rec = pd.read_csv(path + '/Metro_testA/testA_record_2019-01-28.csv')
test_a_rec['startTime'] = test_a_rec['time'].apply(lambda x: x[:15].replace('28', '29') + '0:00')
result = test_a_rec.groupby(['stationID', 'startTime']).status.agg(['count', 'sum']).reset_index()
test = test.merge(result, 'left', ['stationID', 'startTime'])
test['inNums'] = test['sum']
test['outNums'] = test['count'] - test['sum']
test.fillna(0, inplace=True)
test[['stationID', 'startTime', 'endTime', 'inNums', 'outNums']].to_csv(path + '/sub/sub.csv', index=False)
