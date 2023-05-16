import paddle

class MultiTaskLSTM(paddle.nn.Layer):
    """多任务LSTM时序预测模型
    LSTM为共享层网络，对两个预测目标分别有两个分支独立线性层网络
    
    TODO 其实该模型就是个Encoder，如果后续要引入天气预测未来的变量，补充个Decoder，
    然后Encoder负责历史变量的编码，Decoder负责将 编码后的历史编码结果 和 它编码未来变量的编码结果 合并后，做解码预测即可
    """
    def __init__(self,feat_num=14, hidden_size=64, num_layers=2, dropout_rate=0.7, input_len=120*4, pred_len=24*4):
        super(MultiTaskLSTM, self).__init__()
        # LSTM为共享层网络
        self.lstm_layer = paddle.nn.LSTM(feat_num, hidden_size, 
                                    num_layers=num_layers, 
                                    direction='forward', 
                                    dropout=dropout_rate)
        # 为'ROUND(A.POWER,0)'构建分支网络
        self.linear1_1 = paddle.nn.Linear(in_features=input_len*hidden_size, out_features=hidden_size*2)
        self.linear1_2 = paddle.nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.linear1_3 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
        # 为'YD15'构建分支网络 
        self.linear2_1 = paddle.nn.Linear(in_features=input_len*hidden_size, out_features=hidden_size*2)
        self.linear2_2 = paddle.nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.linear2_3 = paddle.nn.Linear(in_features=hidden_size, out_features=pred_len)
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
        output1 = self.dropout(output1)
        output1 = self.linear1_3(output1)

        output2 = self.linear2_1(output)
        output2 = self.dropout(output2)
        output2 = self.linear2_2(output2)
        output2 = self.dropout(output2)
        output2 = self.linear2_3(output2)

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