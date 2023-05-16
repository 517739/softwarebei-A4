# %matplotlib inline
import os
import datetime
import pickle
import paddle
import paddle.fluid as fluid
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 随机种子，保证实验能复现
import random
seed = 42
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

import warnings
warnings.filterwarnings('ignore')

def data_preprocess(df):
    """数据预处理：
        1、读取数据
        2、数据排序
        3、去除重复值
        4、重采样（可选）
        5、缺失值处理
        6、异常值处理
    """
    # ===========读取数据===========
    df = df.sort_values(by='DATATIME', ascending=True)
    print('df.shape:', df.shape)
    print(f"Time range from {df['DATATIME'].values[0]} to {df['DATATIME'].values[-1]}")

    # ===========去除重复值===========
    df = df.drop_duplicates(subset='DATATIME', keep='first')
    print('After Dropping dulicates:', df.shape)

    # ===========重采样（可选） + 线性插值===========
    df = df.set_index('DATATIME')
    # 重采样（可选）：比如04风机缺少2022-04-10和2022-07-25两天的数据，重采样会把这两天数据补充进来
    # df = df.resample(rule=to_offset('15T').freqstr, label='right', closed='right').interpolate(method='linear', limit_direction='both').reset_index()
    # TODO 尝试一些其他缺失值处理方式，比如，用同时刻附近风机的值求均值填补缺失值
    df = df.interpolate(method='linear', limit_direction='both').reset_index()
    print('After Resampling:', df.shape)

    # ===========异常值处理===========
    # 当实际风速为0时，功率置为0
    df.loc[df['ROUND(A.WS,1)']==0, 'YD15'] = 0

    # TODO 风速过大但功率为0的异常：先设计函数拟合出：实际功率=f(风速)，
    # 然后代入异常功率的风速获取理想功率，替换原异常功率

    # TODO 对于在特定风速下的离群功率（同时刻用IQR检测出来），做功率修正（如均值修正）
    return df

def feature_engineer(df):
    """特征工程：时间戳特征"""
    # 时间戳特征
    df['month'] = df.DATATIME.apply(lambda row: row.month, 1)
    df['day'] = df.DATATIME.apply(lambda row: row.day, 1)
    df['weekday'] = df.DATATIME.apply(lambda row: row.weekday(), 1)
    df['hour'] = df.DATATIME.apply(lambda row: row.hour, 1)
    df['minute'] = df.DATATIME.apply(lambda row: row.minute, 1)

    # TODO 挖掘更多特征：差分序列、同时刻风场/邻近风机的特征均值/标准差等
    return df

# 早停
class EarlyStopping():
    """早停
    当验证集超过patience个epoch没有出现更好的评估分数，及早终止训练
    若当前epoch表现超过历史最佳分数，保存该节点模型
    参考：https://blog.csdn.net/m0_63642362/article/details/121244655
    """
    def __init__(self, patience=7, verbose=False, delta=0, ckp_save_path='/home/aistudio/submission/model/model_checkpoint_windid_04.pdparams'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.ckp_save_path = ckp_save_path

    def __call__(self, val_loss, model):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        # 首轮，直接更新best_score和保存节点模型
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 若当前epoch表现没超过历史最佳分数，且累积发生次数超过patience，早停
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        # 若当前epoch表现超过历史最佳分数，更新best_score，保存该节点模型
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # 保存模型
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        paddle.save(model.state_dict(), self.ckp_save_path)
        self.val_loss_min = val_loss

# 数据集划分
# unix时间戳转换
def to_unix_time(dt):
    # timestamp to unix
    epoch = datetime.datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds())


def from_unix_time(unix_time):
    # unix to timestamp
    return datetime.datetime.utcfromtimestamp(unix_time)


class TSDataset(paddle.io.Dataset):
    """时序DataSet
    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    """

    def __init__(self, data,
                 ts_col='DATATIME',
                 use_cols=['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                           'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                           'month', 'day', 'weekday', 'hour', 'minute'],
                 labels=['ROUND(A.POWER,0)', 'YD15'],
                 input_len=24 * 4 * 5, pred_len=24 * 4, stride=19 * 4, data_type='train',
                 train_ratio=0.7, val_ratio=0.15):
        super(TSDataset, self).__init__()
        self.ts_col = ts_col  # 时间戳列
        self.use_cols = use_cols  # 训练时使用的特征列
        self.labels = labels  # 待预测的标签列
        self.input_len = input_len  # 模型输入数据的样本点长度，15分钟间隔，一个小时14个点，近5天的数据就是24*4*5
        self.pred_len = pred_len  # 预测长度，预测次日00:00至23:45实际功率，即1天：24*4
        self.data_type = data_type  # 需要加载的数据类型
        self.scale = True  # 是否需要标准化
        self.train_ratio = train_ratio  # 训练集划分比例
        self.val_ratio = val_ratio  # 验证集划分比例
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率
        # 所以x和label要间隔19*4个点
        self.stride = stride
        assert data_type in ['train', 'val', 'test']  # 确保data_type输入符合要求
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[self.data_type]

        self.transform(data)

    def transform(self, df):
        # 获取unix时间戳、输入特征和预测标签
        time_stamps, x_values, y_values = df[self.ts_col].apply(lambda x: to_unix_time(x)).values, df[
            self.use_cols].values, df[self.labels].values
        # 划分数据集
        # 这里可以按需设置划分比例
        num_train = int(len(df) * self.train_ratio)
        num_vali = int(len(df) * self.val_ratio)
        num_test = len(df) - num_train - num_vali
        border1s = [0, num_train - self.input_len - self.stride, len(df) - num_test - self.input_len - self.stride]
        border2s = [num_train, num_train + num_vali, len(df)]
        # 获取data_type下的左右数据截取边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 使用训练集得到scaler对象
            train_data = x_values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(x_values)
            # 保存scaler
            pickle.dump(self.scaler, open('/home/aistudio/submission/model/scaler.pkl', 'wb'))
        else:
            data = x_values

        # array to paddle tensor
        self.time_stamps = paddle.to_tensor(time_stamps[border1:border2], dtype='int64')
        self.data_x = paddle.to_tensor(data[border1:border2], dtype='float32')
        self.data_y = paddle.to_tensor(y_values[border1:border2], dtype='float32')

    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率
        # 所以x和label要间隔19*4个点
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end + self.stride
        r_end = r_begin + self.pred_len

        # TODO 可以增加对未来可见数据的获取
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        ts_x = self.time_stamps[s_begin:s_end]
        ts_y = self.time_stamps[r_begin:r_end]
        return seq_x, seq_y, ts_x, ts_y

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.data_x) - self.input_len - self.stride - self.pred_len + 1


class TSPredDataset(paddle.io.Dataset):
    """时序Pred DataSet
    划分数据集、适配dataloader所需的dataset格式
    ref: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
    """

    def __init__(self, data,
                 ts_col='DATATIME',
                 use_cols=['WINDSPEED', 'PREPOWER', 'WINDDIRECTION', 'TEMPERATURE', 'HUMIDITY',
                           'PRESSURE', 'ROUND(A.WS,1)', 'ROUND(A.POWER,0)', 'YD15',
                           'month', 'day', 'weekday', 'hour', 'minute'],
                 labels=['ROUND(A.POWER,0)', 'YD15'],
                 input_len=24 * 4 * 5, pred_len=24 * 4, stride=19 * 4):
        super(TSPredDataset, self).__init__()
        self.ts_col = ts_col  # 时间戳列
        self.use_cols = use_cols  # 训练时使用的特征列
        self.labels = labels  # 待预测的标签列
        self.input_len = input_len  # 模型输入数据的样本点长度，15分钟间隔，一个小时14个点，近5天的数据就是24*4*5
        self.pred_len = pred_len  # 预测长度，预测次日00:00至23:45实际功率，即1天：24*4
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率
        # 所以x和label要间隔19*4个点
        self.stride = stride
        self.scale = True  # 是否需要标准化

        self.transform(data)

    def transform(self, df):
        # 获取unix时间戳、输入特征和预测标签
        time_stamps, x_values, y_values = df[self.ts_col].apply(lambda x: to_unix_time(x)).values, df[
            self.use_cols].values, df[self.labels].values
        # 截取边界
        border1 = len(df) - self.input_len - self.stride - self.pred_len
        border2 = len(df)

        # 标准化
        self.scaler = StandardScaler()
        if self.scale:
            # 读取预训练好的scaler
            self.scaler = pickle.load(open('/home/aistudio/submission/model/scaler.pkl', 'rb'))
            data = self.scaler.transform(x_values)
        else:
            data = x_values

        # array to paddle tensor
        self.time_stamps = paddle.to_tensor(time_stamps[border1:border2], dtype='int64')
        self.data_x = paddle.to_tensor(data[border1:border2], dtype='float32')
        self.data_y = paddle.to_tensor(y_values[border1:border2], dtype='float32')

    def __getitem__(self, index):
        """
        实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        # 由于赛题要求利用当日05:00之前的数据，预测次日00:00至23:45实际功率
        # 所以x和label要间隔19*4个点
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end + self.stride
        r_end = r_begin + self.pred_len

        # TODO 可以增加对未来可见数据的获取
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        ts_x = self.time_stamps[s_begin:s_end]
        ts_y = self.time_stamps[r_begin:r_end]
        return seq_x, seq_y, ts_x, ts_y

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.data_x) - self.input_len - self.stride - self.pred_len + 1

# 模型结构定义
class MultiTaskLSTM(paddle.nn.Layer):
    """多任务LSTM时序预测模型
    LSTM为共享层网络，对两个预测目标分别有两个分支独立线性层网络

    TODO 其实该模型就是个Encoder，如果后续要引入天气预测未来的变量，补充个Decoder，
    然后Encoder负责历史变量的编码，Decoder负责将 编码后的历史编码结果 和 它编码未来变量的编码结果 合并后，做解码预测即可
    """

    def __init__(self, feat_num=14, hidden_size=64, num_layers=2, dropout_rate=0.7, input_len=120 * 4, pred_len=24 * 4):
        super(MultiTaskLSTM, self).__init__()
        # LSTM为共享层网络
        self.lstm_layer = paddle.nn.LSTM(feat_num, hidden_size,
                                         num_layers=num_layers,
                                         direction='forward',
                                         dropout=dropout_rate)
        # 为'ROUND(A.POWER,0)'构建分支网络
        self.linear1_1 = paddle.nn.Linear(in_features=input_len * hidden_size, out_features=hidden_size)
        self.linear1_2 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
        # self.linear1_3 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
        # 为'YD15'构建分支网络
        self.linear2_1 = paddle.nn.Linear(in_features=input_len * hidden_size, out_features=hidden_size)
        self.linear2_2 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
        # self.linear2_3 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
        self.dropout = paddle.nn.Dropout(dropout_rate)

    def forward(self, x):
        # x形状大小为[batch_size, input_len, feature_size]
        # output形状大小为[batch_size, input_len, hidden_size]
        # hidden形状大小为[num_layers, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm_layer(x)
        # output: [batch_size, input_len, hidden_size] -> [batch_size, input_len*hidden_size]
        output = paddle.reshape(output, [len(output), -1])

        output1 = self.linear1_1(output)
        output1 = self.dropout(output1)
        output1 = self.linear1_2(output1)
        # output1 = self.dropout(output1)
        # output1 = self.linear1_3(output1)

        output2 = self.linear2_1(output)
        output2 = self.dropout(output2)
        output2 = self.linear2_2(output2)
        # output2 = self.dropout(output2)
        # output2 = self.linear2_3(output2)

        # outputs: ([batch_size, pre_len, 1], [batch_size, pre_len, 1])
        return [output1, output2]

class MultiTaskMSELoss(paddle.nn.Layer):
    """
    设置损失函数, 多任务模型，两个任务MSE的均值做loss输出
    """
    def __init__(self):
        super(MultiTaskMSELoss, self).__init__()

    def forward(self, inputs, labels):
        mse_loss = paddle.nn.loss.MSELoss()
        mse1 = mse_loss(inputs[0], labels[:,:,0].squeeze(-1))
        mse2 = mse_loss(inputs[1], labels[:,:,1].squeeze(-1))
        # TODO 也可以自行设置各任务的权重，让其更偏好YD15
        # 即让多任务有主次之分
        return mse1, mse2, (mse1 + mse2) / 2

def calc_acc(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return 1 - rmse/201000

# 模型训练
def train(df, turbine_id):
    # 设置数据集
    train_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='train')
    val_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='val')
    test_dataset = TSDataset(df, input_len=input_len, pred_len=pred_len, data_type='test')
    print(f'LEN | train_dataset:{len(train_dataset)}, val_dataset:{len(val_dataset)}, test_dataset:{len(test_dataset)}')

    # 设置数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = paddle.io.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset, shuffle=False, batch_size=1, drop_last=False)

    # 设置模型
    model = MultiTaskLSTM()

    # 设置优化器
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=learning_rate, factor=0.5, patience=3, verbose=True)
    opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

    # 设置损失
    mse_loss = MultiTaskMSELoss()

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   ckp_save_path=f'/home/aistudio/submission/model/model_checkpoint_windid_{turbine_id}.pdparams')

    for epoch in tqdm(range(epoch_num)):
        # =====================train============================
        train_epoch_loss, train_epoch_mse1, train_epoch_mse2 = [], [], []
        model.train()  # 开启训练
        for batch_id, data in enumerate(train_loader()):
            x = data[0]
            y = data[1]
            # 预测
            outputs = model(x)
            # 计算损失
            mse1, mse2, avg_loss = mse_loss(outputs, y)
            # 反向传播
            avg_loss.backward()
            # 梯度下降
            opt.step()
            # 清空梯度
            opt.clear_grad()
            train_epoch_loss.append(avg_loss.numpy()[0])
            train_loss.append(avg_loss.item())
            train_epoch_mse1.append(mse1.item())
            train_epoch_mse2.append(mse2.item())
        train_epochs_loss.append(np.average(train_epoch_loss))
        print("epoch={}/{} of train | loss={}, MSE of ROUND(A.POWER,0):{}, MSE of YD15:{} ".format(epoch, epoch_num,
                                                                                                   np.average(
                                                                                                       train_epoch_loss),
                                                                                                   np.average(
                                                                                                       train_epoch_mse1),
                                                                                                   np.average(
                                                                                                       train_epoch_mse2)))

        # =====================valid============================
        model.eval()  # 开启评估/预测
        valid_epoch_loss, valid_epochs_mse1, valid_epochs_mse2 = [], [], []
        for batch_id, data in enumerate(val_loader()):
            x = data[0]
            y = data[1]
            outputs = model(x)
            mse1, mse2, avg_loss = mse_loss(outputs, y)
            valid_epoch_loss.append(avg_loss.numpy()[0])
            valid_loss.append(avg_loss.numpy()[0])
            valid_epochs_mse1.append(mse1.item())
            valid_epochs_mse2.append(mse2.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        print('Valid: MSE of ROUND(A.POWER,0):{}, MSE of YD15:{}'.format(np.average(train_epoch_mse1),
                                                                         np.average(train_epoch_mse2)))

        # ==================early stopping======================
        early_stopping(valid_epochs_loss[-1], model=model)
        if early_stopping.early_stop:
            print(f"Early stopping at Epoch {epoch - patience}")
            break

    print('Train & Valid: ')
    plt.figure(figsize=(12, 3))
    plt.subplot(121)
    plt.plot(train_loss[:], label="train")
    plt.title("train_loss")
    plt.xlabel('iteration')
    plt.subplot(122)
    plt.plot(train_epochs_loss[1:], '-o', label="train")
    plt.plot(valid_epochs_loss[1:], '-o', label="valid")
    plt.title("epochs_loss")
    plt.xlabel('epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =====================test============================
    # 加载最优epoch节点下的模型
    model = MultiTaskLSTM()
    model.set_state_dict(paddle.load(f'/home/aistudio/submission/model/model_checkpoint_windid_{turbine_id}.pdparams'))

    model.eval()  # 开启评估/预测
    test_loss, test_epoch_mse1, test_epoch_mse2 = [], [], []
    test_accs1, test_accs2 = [], []
    for batch_id, data in tqdm(enumerate(test_loader())):
        x = data[0]
        y = data[1]
        ts_y = [from_unix_time(x) for x in data[3].numpy().squeeze(0)]
        outputs = model(x)
        mse1, mse2, avg_loss = mse_loss(outputs, y)
        acc1 = calc_acc(y.numpy().squeeze(0)[:, 0], outputs[0].numpy().squeeze(0))
        acc2 = calc_acc(y.numpy().squeeze(0)[:, 1], outputs[1].numpy().squeeze(0))
        test_loss.append(avg_loss.numpy()[0])
        test_epoch_mse1.append(mse1.numpy()[0])
        test_epoch_mse2.append(mse2.numpy()[0])
        test_accs1.append(acc1)
        test_accs2.append(acc2)

    print('Test: ')
    print('MSE of ROUND(A.POWER,0):{}, MSE of YD15:{}'.format(np.average(test_epoch_mse1), np.average(test_epoch_mse2)))
    print('Mean MSE:', np.mean(test_loss))
    print('ACC of ROUND(A.POWER,0):{}, ACC of YD15:{}'.format(np.average(test_accs1), np.average(test_accs2)))

data_path = '/home/aistudio/功率预测竞赛赛题与数据集'
files = os.listdir(data_path)
debug = True # 为了快速跑通代码，可以先尝试用采样数据做debug

# 遍历每个风机的数据做训练、验证和测试
for f in files:
    df = pd.read_csv(os.path.join(data_path, f),
                    parse_dates=['DATATIME'],
                    infer_datetime_format=True,
                    dayfirst=True)
    turbine_id = int(float(f.split('.csv')[0]))
    print(f'turbine_id:{turbine_id}')

    if debug:
        df = df.iloc[-24*4*200:,:]

    # 数据预处理
    df = data_preprocess(df)
    # 特征工程
    df = feature_engineer(df)

    # 模型参数
    input_len = 120*4     # 输入序列的长度为 120*4
    pred_len = 24*4       #  预测序列的长度为 24*4
    epoch_num = 100       #  模型训练的轮数
    batch_size = 512      # 每个训练批次使用的样本数量
    learning_rate = 0.001 # 学习率
    patience = 10         # 如果连续patience个轮次性能没有提升，就会停止训练。

    # 训练模型
    train(df, turbine_id)

# 预测
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
    model.set_state_dict(paddle.load(f'submission/model/model_checkpoint_windid_{turbine_id}.pdparams'))
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

files = os.listdir('infile')
if not os.path.exists('pred'):
    os.mkdir('pred')
# 第一步，完成数据格式统一
for f in files:
    if '.csv' not in f:
        continue
    print(f)
    # 获取文件路径
    data_file = os.path.join('infile', f)
    print(data_file)
    out_file = os.path.join('pred', f[:4] + 'out.csv')
    df = pd.read_csv(data_file,
                    parse_dates=['DATATIME'],
                    infer_datetime_format=True,
                    dayfirst=True)
    turbine_id = df.TurbID[0]
    # 预测结果
    forecast(df, turbine_id, out_file)