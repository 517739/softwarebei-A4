import shutil
import os
import joblib
import pandas as pd
import paddle
from paddle import nn


def load_models():
    model_dict = paddle.load("model/" + str(data.TurbID.values[0]) + ".pdparams")
    model1 = GCN(input_size=7, hidden_size=64, output_size=1)
    model2 = joblib.load("model/gbm" + str(data.TurbID.values[0]) + '.pkl')
    model1.set_state_dict(model_dict)
    # 开启评估模式
    model1.eval()
    return [model1, model2]


def weightPredict(models):
    with paddle.no_grad():
        adj = paddle.ones((train.shape[0], train.shape[0]))  # 假设所有节点之间都有连接
        output1 = models[0](paddle.to_tensor(train.values, dtype="float32"), adj).numpy().flatten()
        print(output1)
        output2 = models[1].predict(train)
        output = (-0.01 * output1 + 0.99 * output2)
        datas = {"TurbID": new_data["TurbID"], "DATATIME": new_data1['DATATIME'],
                 "ROUND(A.POWER,0)": new_data['ROUND(A.POWER,0)'],
                 "YD15": output.flatten()}
        frame = pd.DataFrame(datas)
        return frame


# 定义GCN模型
class GCN(nn.Layer):

    def __init__(self, input_size, hidden_size, output_size):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = paddle.matmul(adj, x)
        x = self.linear1(x)
        x = self.relu(x)
        x = paddle.matmul(adj, x)
        x = self.linear2(x)
        return x


# 定义数据集
class GraphDataset(paddle.io.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


folder_path = "./pred"
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.mkdir(folder_path)
csv_files = [file for file in os.listdir("./infile") if file.endswith(".csv")]
for i in range(1, len(csv_files) + 1):
    n = 4 - len(str(i))
    filename = '0' * n + str(i) + 'in.csv'
    outname = '0' * n + str(i) + 'out.csv'
    data = pd.read_csv("./infile/" + filename)
    data = data.fillna(0)
    key = str(data['DATATIME'].max())[:11]
    new_data = data[data['DATATIME'].str.startswith(key)]
    new_data1 = new_data.copy()
    train = new_data.drop(["TurbID", "DATATIME", "ROUND(A.POWER,0)", 'YD15'], axis=1)
    models = load_models()
    frame = weightPredict(models)
    frame.to_csv("pred/" + outname, index=False)
