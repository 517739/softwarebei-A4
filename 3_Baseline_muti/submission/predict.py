#导入需要的包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
import paddlets
from paddlets import TSDataset
from paddlets import TimeSeries
from paddlets.models.forecasting import MLPRegressor, LSTNetRegressor
from paddlets.transform import Fill, StandardScaler
from paddlets.metrics import MSE, MAE
from paddlets.analysis import AnalysisReport, Summary
from paddlets.datasets.repository import get_dataset
from paddlets.models.model_loader import load
import warnings
warnings.filterwarnings('ignore')

def forecast_(df, turbine_id, out_file):
    test_dataset = TSDataset.load_from_dataframe(
        df,
        time_col='DATATIME',
        target_cols=['YD15'],
        observed_cov_cols=[
            'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE',
            'HUMIDITY', 'PRESSURE'
        ],
        freq='15min',
        fill_missing_dates=True,
        fillna_method='pre')
    # scaler = StandardScaler()
    # scaler.fit(test_dataset)
    # test_dataset_scaled = scaler.transform(test_dataset)
    # 模型加载
    load_dir = 'model/' + turbine_id
    loaded_model0 = load(load_dir + '/paddlets-ensemble-model0')
    loaded_model1 = load(load_dir + '/paddlets-ensemble-model1')
    loaded_model2 = load(load_dir + '/paddlets-ensemble-model2')
    # 模型预测
    result = loaded_model0.predict(test_dataset).to_dataframe()[19*4:]*0.2 + \
             loaded_model1.predict(test_dataset).to_dataframe()[19*4:]*0.4 + \
             loaded_model2.predict(test_dataset).to_dataframe()[19*4:]*0.4
    # 获取预测数据
    # result = result.to_dataframe()[19 * 4:]
    result = result.reset_index()
    # 传入风场风机ID
    result['TurbID'] = turbine_id
    # 重新调整字段名称和顺序
    result.rename(columns={"index": "DATATIME"}, inplace=True)
    result = result[['TurbID', 'DATATIME', 'YD15']]
    result.to_csv(out_file, index=False)

def forecast(df, turbine_id, out_file):
    test_dataset = TSDataset.load_from_dataframe(
        df,
        time_col='DATATIME',
        target_cols=['ROUND(A.POWER,0)', 'YD15'],
        observed_cov_cols=[
            'WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE',
            'HUMIDITY', 'PRESSURE', 'ROUND(A.WS,1)'
        ],
        freq='15min',
        fill_missing_dates=True,
        fillna_method='pre')
    # scaler = StandardScaler()
    # scaler.fit(test_dataset)
    # test_dataset_scaled = scaler.transform(test_dataset)
    # 模型加载
    load_dir = 'model/' + turbine_id
    loaded_model0 = load(load_dir + '/paddlets-ensemble-model0')
    loaded_model1 = load(load_dir + '/paddlets-ensemble-model1')
    loaded_model2 = load(load_dir + '/paddlets-ensemble-model2')
    # 模型预测
    result = loaded_model0.predict(test_dataset).to_dataframe()[19*4:]*0.2 + \
             loaded_model1.predict(test_dataset).to_dataframe()[19*4:]*0.4 + \
             loaded_model2.predict(test_dataset).to_dataframe()[19*4:]*0.4
    # 获取预测数据
    # result = result.to_dataframe()[19 * 4:]
    result = result.reset_index()
    # 传入风场风机ID
    result['TurbID'] = turbine_id
    # 重新调整字段名称和顺序
    result.rename(columns={"index": "DATATIME"}, inplace=True)
    result = result[['TurbID', 'DATATIME', 'ROUND(A.POWER,0)', 'YD15']]
    result.to_csv(out_file, index=False)

if __name__=="__main__":

    files = os.listdir('infile')
    os.mkdir('pred')
    # 第一步，完成数据格式统一
    for f in files:
        print(f)
        # 获取文件路径
        data_file = os.path.join('infile', f)
        print(data_file)
        out_file = os.path.join('pred', f[:4] + 'out.csv')
        # print(out_file)
        df = pd.read_csv(data_file,
                        parse_dates=['DATATIME'],
                        infer_datetime_format=True,
                        dayfirst=True,
                        dtype={
                            'WINDDIRECTION': np.float64,
                            'HUMIDITY': np.float64,
                            'PRESSURE': np.float64
                        })
        # 因为数据批次不同，数据集中有一些时间戳重复的脏数据，送入paddlets前要进行处理，本赛题要求保留第一个数据
        df = df.drop_duplicates(subset=['DATATIME'], keep='first')
        # 获取风机号
        turbine_id = df.TurbID[0]
        if turbine_id < 10:
            turbine_id = '0' + str(turbine_id)
        else:
            turbine_id = str(turbine_id)
        print(turbine_id)
        df = df.drop(['TurbID'], axis=1) #按列删
        # 裁剪倒数第二天5:00前的数据输入时间序列
        tail = df.tail(4 * (19 + 24)).index
        df = df.drop(tail)
        if turbine_id in ["02","07","08"]:
            forecast_(df, turbine_id, out_file)
        else:
            forecast(df, turbine_id, out_file)
