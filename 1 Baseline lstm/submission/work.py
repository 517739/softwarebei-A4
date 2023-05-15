import numpy as np
import pandas as pd

for id in range(1,11):
    # 读取一份数据文件，这里需要特别注意时间戳字段的格式指定
    data_dir = '功率预测竞赛赛题与数据集/' + id + '.csv'
    print(type(data_dir))
    df = pd.read_csv(data_dir, parse_dates=['DATATIME'], infer_datetime_format=True, dayfirst=True)

    # 因为数据批次不同，数据集中有一些时间戳重复的脏数据，送入paddlets前要进行处理，本赛题要求保留第一个数据
    df.drop_duplicates(subset=['DATATIME'], keep='first', inplace=True)

    if flag == 1:
        target_cov_dataset = TSDataset.load_from_dataframe(
            df,
            time_col='DATATIME',
            target_cols='YD15',
            observed_cov_cols=['WINDSPEED', 'PREPOWER', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)'],
            freq='15min',
            fill_missing_dates=True,
            fillna_method='pre'
        )

    # 数据集划分
    train_dataset, val_test_dataset = target_cov_dataset.split(0.7)
    val_dataset, test_dataset = val_test_dataset.split(0.3)
    train_dataset.plot(add_data=[val_dataset, test_dataset], labels=['Val', 'Test'])

    # 归一化
    scaler = StandardScaler()
    scaler.fit(train_dataset)
    train_dataset_scaled = scaler.transform(train_dataset)
    val_test_dataset_scaled = scaler.transform(val_test_dataset)
    val_dataset_scaled = scaler.transform(val_dataset)
    test_dataset_scaled = scaler.transform(test_dataset)
    print(train_dataset_scaled)
    print(val_dataset_scaled)

    lstm = LSTNetRegressor(
        in_chunk_len=(24 + 19) * 7 * 4,
        out_chunk_len=(24 + 19) * 4,
        max_epochs=10,
        optimizer_params=dict(learning_rate=5e-3),
    )

    lstm.fit(train_dataset_scaled, val_dataset_scaled)

    # 预测结果并可视化
    subset_test_pred_dataset = lstm.predict(val_dataset)
    subset_test_dataset, _ = test_dataset.split(len(subset_test_pred_dataset.target))
    subset_test_dataset.plot(add_data=subset_test_pred_dataset, labels=['Pred'])

    # 模型评估
    mae = MAE()
    res = mae(subset_test_dataset, subset_test_pred_dataset)
    print(type(res))
    print(res)
    f = open(f'./mae/00{id}.txt', 'w')
    f.write(f'00{id}:\n' + str(res))
    f.close()

    # 模型保存
    lstm.save(f"./lstm_model/{id}")

    # 模型加载
    from paddlets.models.model_loader import load

    loaded_lstm = load(f"./lstm_model/{id}")
    # 模型预测
    result = loaded_lstm.predict(test_dataset)
    result.to_dataframe()[19 * 4:]

    # 截取次日预测数据
    result = result.to_dataframe()[19 * 4:]
    result = result.reset_index()
    # 传入风场风机ID
    result['TurbID'] = 1
    # 重新调整字段名称和顺序
    result.rename(columns={"index": "Datetime"}, inplace=True)
    result = result[['TurbID', 'Datetime', 'YD15']]
    result.to_csv(f'submit/00{id}out.csv', index=False)
