from pandas import read_csv
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
from parser_my import args


def getData(corpusFile, sequence_length, batchSize):
    # 数据预处理 ，去除id、股票代码、前一天的收盘价、交易日期等对训练无用的无效数据
    stock_data = read_csv(corpusFile)  # 从CSV文件中读取股票数据
    stock_data.drop("ts_code", axis=1, inplace=True)  # 删除第二列股票代码
    stock_data.drop("id", axis=1, inplace=True)  # 删除第一列id
    stock_data.drop(
        "pre_close", axis=1, inplace=True
    )  # 删除列pre_close 前一天的收盘价列
    stock_data.drop("trade_date", axis=1, inplace=True)  # 删除第三列trade_date 交易日期

    close_max = stock_data["close"].max()  # 收盘价的最大值
    close_min = stock_data["close"].min()  # 收盘价的最小值
    df = stock_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))  # 归一化

    # 构造特征和标签
    # 根据前n天的数据，预测未来一天的收盘价(close)， 例如：根据1月1日、1月2日、1月3日、1月4日、1月5日的数据（每一天的数据包含8个特征），预测1月6日的收盘价。
    sequence = sequence_length  # 天数
    X = []  # 特征
    Y = []  # 标签
    for i in range(df.shape[0] - sequence):
        X.append(
            np.array(df.iloc[i : (i + sequence),].values, dtype=np.float32)
        )  # 构造特征  取从第 i 天开始的sequence天的数据
        Y.append(
            np.array(df.iloc[(i + sequence), 0], dtype=np.float32)
        )  # 构造标签  取第 i + sequence 天的收盘价

    # 构建训练集和测试集
    total_len = len(Y)  # 总数据量
    # print(total_len)
    trainx, trainy = X[: int(0.99 * total_len)], Y[: int(0.99 * total_len)]  # 训练集
    testx, testy = X[int(0.99 * total_len) :], Y[int(0.99 * total_len) :]  # 测试集
    # 构建DataLoader，用于批次训练
    train_loader = DataLoader(
        dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()),
        batch_size=batchSize,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True
    )  # 测试集数据加载器
    return (
        close_max,
        close_min,
        train_loader,
        test_loader,
    )  # 返回最大值、最小值、训练集和测试集数据加载器


class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx  # 特征数据
        self.y = yy  # 标签数据
        self.tranform = transform  # 数据变换（如果有的话）

    def __getitem__(self, index):
        x1 = self.x[index]  # 获取指定索引的特征数据
        y1 = self.y[index]  # 获取指定索引的标签数据
        if self.tranform != None:
            return self.tranform(x1), y1  # 如果有数据变换，则应用变换
        return x1, y1  # 否则，直接返回数据

    def __len__(self):
        return len(self.x)  # 返回数据集的长度
