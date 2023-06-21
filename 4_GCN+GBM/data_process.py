import os
import sys
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt

def all_data_process(data_dir):
    files = os.listdir(data_dir)
    for f in files:
        # 获取文件路径
        data_file = os.path.join(data_dir, f)
        print(data_file)
        # 获取文件名后缀(区域赛训练集\11)
        data_type = os.path.splitext(data_file)[-1]
        # 获取文件名前缀(.csv)
        data_name = os.path.splitext(data_file)[0]
        wind_id = data_name[-2:]

        # 读入
        df = pd.read_csv(data_file, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)

        # 处理
        # 加参数
        # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
        # df['diff_YD'] = df['YD15'].diff()
        # df['diff_YD'].iloc[0] = 0.0

        # ===========大空白筛除==========
        # df = df.iloc[22082:, :]
        # df.dropna(subset=["YD15"], inplace=True)

        # ===========去除重复值===========
        # print('Before Dropping dulicates:', data.shape)
        df = df.drop_duplicates(subset='DATATIME', keep='first')
        # print('After Dropping dulicates:', data.shape)

        # ===========重采样（可选） + 线性插值===========
        df['DATATIME'] = pd.to_datetime(df['DATATIME'])
        df = df.set_index('DATATIME')  # 用datatime作为行索引
        # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
        df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
        # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
        df = df.interpolate(method='linear', limit_direction='both').reset_index()
        # data = data.reset_index()
        # print('After Resampling:', data.shape)
        # print(data)
        # print(type(data))

        # ===========异常值处理===========
        df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
        df.loc[df['WINDSPEED'] >= 60, 'YD15'] = np.nan
        df.loc[df['PREPOWER'] == 0, 'YD15'] = np.nan
        # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan
        print(df)

        # ===========空值填补===========
        # print('Before:', data.shape)
        df.dropna(subset=["YD15"], inplace=True)
        df_median = df.median()
        df_mean = df.mean(axis=0)
        df = df.fillna(value=df_median)
        # print('After:', data.shape)
        new_data = df.copy()

        # ===========画图===========
        new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12, 20))
        plt.show()

        # 输出
        df.to_csv('处理后数据/' + wind_id + '.csv', encoding='utf-8')

def sigmoid(x):
    # sigmoid函数式，返回的是数组
    return 1/(1+np.exp(-x))

def div(WD,PO):
    WD = WD * WD * WD + 1
    return PO / WD

def data_process_11(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')


def data_process_15(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    df.loc[ : , 'YD15'] = df.loc[ : , 'ROUND(A.POWER,0)']
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')


def data_process_17(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    df.loc[35486:, 'YD15'] = np.nan
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')

def data_process_18(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    df = df.iloc[3073:, :]

    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    df.loc[df['PREPOWER'] == 0, 'YD15'] = np.nan
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan
    print(df)

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')

def data_process_19(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    df = df.iloc[22082:,:]

    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    df.loc[df['PREPOWER'] == 0, 'YD15'] = np.nan
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan
    print(df)

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')

# 16 20
def data_process_20(data_dir):
    # 读入
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
    wind_id = data_dir[7:9]
    print(wind_id)

    # 处理
    # 加参数
    # df['PO/W3']=df.apply(lambda x:div(x['WINDSPEED'],x['PREPOWER']),axis=1)
    # df['diff_YD'] = df['YD15'].diff()
    # df['diff_YD'].iloc[0] = 0.0

    df.dropna(subset=["YD15"], inplace=True)
    # ===========去除重复值===========
    # print('Before Dropping dulicates:', data.shape)
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    # print('After Dropping dulicates:', data.shape)

    # ===========重采样（可选） + 线性插值===========
    df['DATATIME'] = pd.to_datetime(df['DATATIME'])
    df = df.set_index('DATATIME')  # 用datatime作为行索引
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right')
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    # data = data.reset_index()
    # print('After Resampling:', data.shape)
    # print(data)
    # print(type(data))

    # ===========异常值处理===========
    df.loc[df['ROUND(A.WS,1)'] == 0, 'YD15'] = 0
    df.loc[df['WINDSPEED'] >= 60,'YD15'] = np.nan
    df.loc[df['PREPOWER'] == 0, 'YD15'] = np.nan
    # df.loc[df['PO/W3'] > 5000, 'YD15'] = np.nan
    print(df)

    # ===========空值填补===========
    # print('Before:', data.shape)
    df.dropna(subset=["YD15"], inplace=True)
    df_median = df.median()
    df_mean = df.mean(axis=0)
    df = df.fillna(value=df_median)
    # print('After:', data.shape)
    new_data = df.copy()

    # ===========画图===========
    new_data.plot(x='DATATIME', y=new_data.columns[1:], subplots=True, figsize=(12,20))
    plt.show()

    # 输出
    df.to_csv(f'处理后数据/{wind_id}.csv', encoding='utf-8')


if __name__ == '__main__':
    wind_id = 20
    data_dir = f'区域赛训练集/{wind_id}.csv'
    data_process_20(data_dir)


