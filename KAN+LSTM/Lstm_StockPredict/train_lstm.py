from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from parser_my import args
from dataset import getData
import matplotlib.pyplot as plt

def train():
    # 创建lstm模型实例，并将其移动到指定的设备（CPU或GPU）
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    close_max, close_min, train_loader, test_loader = getData(args.corpusFile,args.sequence_length,args.batch_size ) # 获取预处理后的数据
    
    # 初始化列表用于记录每个epoch的总损失
    epoch_losses = []

    # 打开一个名为 'lstm_loss_log.txt' 的文本文件，用于记录每个epoch的损失值
    with open('lstm_loss_log.txt', 'w') as f:
        for i in range(args.epochs):  # 循环遍历每个epoch
            total_loss = 0  # 初始化当前epoch的总损失
            for idx, (data, label) in enumerate(train_loader):
                if args.useGPU: # 如果使用GPU
                    data1 = data.squeeze(1).cuda() # 删除data张量中的第一个维度，并将其移动到GPU
                    pred = model(Variable(data1).cuda()) # 将data1封装成Variable并传入模型进行前向传播，得到预测值
                    # print(pred.shape) 
                    pred = pred[1,:,:]  # 这里取pred的第二个维度的数据作为最终预测结果
                    label = label.unsqueeze(1).cuda()  # 将标签数据添加一个维度并移动到GPU
                    # print(label.shape)
                else:  # 如果使用CPU
                    data1 = data.squeeze(1) # 删除data张量中的第一个维度
                    pred = model(Variable(data1))   # 将data1封装成Variable并传入模型进行前向传播，得到预测值
                    pred = pred[1, :, :]  # 这里取pred的第二个维度的数据作为最终预测结果
                    label = label.unsqueeze(1) # 将标签数据添加一个维度
                loss = criterion(pred, label) # 计算当前batch的损失值
                optimizer.zero_grad() # 清空优化器的梯度
                loss.backward()   # 反向传播，计算梯度
                optimizer.step() # 更新模型参数
                total_loss += loss.item()   # 累加当前batch的损失值到total_loss
            
            # 记录每个epoch的总损失
            epoch_losses.append(total_loss)
            
            # 在终端输出第多少轮和对应的loss
            print(f'Epoch {i+1}, Loss: {total_loss}')
            
            # 将损失写入文件
            f.write(f'Epoch {i+1}, Loss: {total_loss}\n')
            
            if i % 10 == 0: # 每10个epoch保存一次模型
                torch.save({'state_dict': model.state_dict()}, args.save_file) # 保存模型的状态字典到指定文件
                print('第%d epoch，保存模型' % i) # 打印当前epoch信息，表示模型已经保存
        
        torch.save({'state_dict': model.state_dict()}, args.save_file) # 在训练结束后，保存最终模型

train()
