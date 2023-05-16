import os
import paddle
import pandas as pd
from env.data_preprocess import data_preprocess, feature_engineer
from env.data_loader import TSDataset, TSPredDataset
from env.model import MultiTaskLSTM, MultiTaskMSELoss
from env.utils import EarlyStopping, calc_acc, to_unix_time, from_unix_time
import warnings
warnings.filterwarnings('ignore')


def forecast(df, turbine_id, out_file):
    # 数据预处理
    df = data_preprocess(df)
    # 特征工程
    df = feature_engineer(df)
    # 准备数据加载器
    input_len = 120*4
    pred_len = 24*4
    pred_dataset = TSPredDataset(df, input_len = input_len, pred_len = pred_len)
    pred_loader = paddle.io.DataLoader(pred_dataset, shuffle=False, batch_size=1, drop_last=False)
    # 定义模型
    model = MultiTaskLSTM()
    # 导入模型权重文件
    model.set_state_dict(paddle.load(f'model/model_checkpoint_windid_{turbine_id}.pdparams'))
    model.eval() # 开启预测
    for batch_id, data in enumerate(pred_loader()):
        x = data[0]
        y = data[1]
        outputs = model(x)
        round = [x for x in outputs[0].numpy().squeeze()]
        yd15 = [x for x in outputs[1].numpy().squeeze()]
        ts_x = [from_unix_time(x) for x in data[2].numpy().squeeze(0)]
        ts_y = [from_unix_time(x) for x in data[3].numpy().squeeze(0)]

    result = pd.DataFrame({'DATATIME':ts_y, 'ROUND(A.POWER,0)':round, 'YD15':yd15})
    result['TurbID'] = turbine_id
    result = result[['TurbID', 'DATATIME', 'ROUND(A.POWER,0)', 'YD15']]
    result.to_csv(out_file, index=False)
    

if __name__=="__main__":
    files = os.listdir('infile')
    if not os.path.exists('pred'):
        os.mkdir('pred')
    # 第一步，完成数据格式统一
    for f in files:
        if '.csv' not in f:
            continue
        # print(f)
        # 获取文件路径
        data_file = os.path.join('infile', f)
        # print(data_file)
        out_file = os.path.join('pred', f[:4] + 'out.csv')
        df = pd.read_csv(data_file,
                        parse_dates=['DATATIME'],
                        infer_datetime_format=True,
                        dayfirst=True)
        turbine_id = df.TurbID[0]
        # 预测结果
        forecast(df, turbine_id, out_file)