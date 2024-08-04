from LSTMModel import lstm  # 导入自定义的LSTM模型类
from dataset import getData  # 导入数据处理函数
from parser_my import args  # 导入命令行参数解析模块
import torch
from torch import nn


def write_to_txt(preds, labels, filename):
    with open(filename, "w") as file:
        for pred, label in zip(preds, labels):
            file.write(f"{pred},{label}\n")


def eval():
    # 加载预训练模型
    # model = torch.load(args.save_file)
    model = lstm(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_size=1,
    )  # 创建LSTM模型实例
    model.to(args.device)  # 将模型移到指定的设备（CPU或GPU）
    checkpoint = torch.load(args.save_file)  # 加载模型的状态字典
    model.load_state_dict(checkpoint["state_dict"])  # 将状态字典加载到模型中
    preds = []  # 初始化预测值列表
    labels = []  # 初始化真实标签值列表
    close_max, close_min, train_loader, test_loader = getData(
        args.corpusFile, args.sequence_length, args.batch_size
    )  # 获取训练和测试数据
    for idx, (x, label) in enumerate(test_loader):  # 遍历测试数据进行预测
        if args.useGPU:
            x = x.squeeze(
                1
            ).cuda()  # batch_size,seq_len,input_size # 如果使用GPU，将数据移到GPU并调整维度
        else:
            x = x.squeeze(1)  # 调整数据维度
        pred = model(x)  # 使用模型进行预测
        list = pred.data.squeeze(1).tolist()  # 将预测结果转换为列表
        preds.extend(list[-1])  # 记录每个样本的最后一个预测值
        # 将预测结果转换为列表，然后将其包装在列表中，再将其添加到 preds 中
        # pred_list = [pred.data.squeeze(1).tolist()]
        # preds.extend(pred_list[-1])  # 记录每个样本的最后一个预测值
        labels.extend(label.tolist())  # 记录真实标签值
    # # 计算评价指标
    rmse = torch.sqrt(
        nn.MSELoss()(torch.tensor(preds), torch.tensor(labels))
    )  # 均方根误差
    mae = torch.mean(
        torch.abs(torch.tensor(preds) - torch.tensor(labels))
    )  # 平均绝对误差

    print(f"RMSE: {rmse.item():.6f}")
    print(f"MAE: {mae.item():.6f}")

    # 将评价指标写入文件
    with open("evaluation_lstm.txt", "w", encoding="utf-8") as f:
        f.write(f"RMSE: {rmse.item():.6f}\n")
        f.write(f"MAE: {mae.item():.6f}\n")
    # 打印预测值和真实值
    # for i in range(len(preds)):
    #     # 将预测值反标准化并打印、将真实值反标准化并打印
    #     print('预测值是%.6f,真实值是%.6f' % (
    #     preds[i][0] * (close_max - close_min) + close_min, labels[i] * (close_max - close_min) + close_min))
    # write_to_txt(preds, labels, "predictions_lstm.txt")


eval()
