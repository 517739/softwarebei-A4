import pandas as pd

wind_id = '07'
data_path = f'Data/{wind_id}.csv'
df = pd.read_csv(data_path, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)
df = df.sort_values(by='DATATIME', ascending=True)
print(df)
df = df.drop(columns='PLANTID', axis=1)  # 7号去掉多余的第一列
df.to_csv(f'{wind_id}.csv',index=0, encoding='utf-8')
print(df.columns)