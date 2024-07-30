from LSTMModel import lstm # 导入自定义的LSTM模型类
from dataset import getData  # 导入数据处理函数
from parser_my import args # 导入命令行参数解析模块
import matplotlib.pyplot as plt

# 获取训练和测试数据
close_max, close_min, train_loader, test_loader = getData(r"data\000001SH_index.csv", 5, 64)

# 打印close_max和close_min，确保它们的值是正确的
print(f"close_max: {close_max}, close_min: {close_min}")

# 读取文件内容并去除方括号
def read_predictions_file(file_path):
    preds, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            pred, label = line.strip().split(',')
            # 去除方括号并转换为浮点数
            pred = float(pred.strip('[]'))
            labels.append(float(label))
            preds.append(pred)
    return preds, labels

# 反标准化
def reverse_standardize(values, close_max, close_min):
    return [value * (close_max - close_min) + close_min for value in values]

# 绘制预测值和真实值
def plot_predictions(preds, labels):
    plt.figure(figsize=(14, 7))  # 设置图像大小
    plt.plot(preds, label='Predicted Prices')
    plt.plot(labels, label='Actual Prices')
    plt.xlabel('Time Step')  # 横轴标签
    plt.ylabel('Stock Price')  # 纵轴标签
    plt.legend()
    plt.title('Predicted vs Actual Stock Prices')  # 图像标题
    plt.savefig("lstm_predict")
    plt.show()

# 主函数
def main():
    file_path = 'predictions_lstm.txt'  # 文件路径

    preds, labels = read_predictions_file(file_path)
    preds = reverse_standardize(preds, close_max, close_min)
    labels = reverse_standardize(labels, close_max, close_min)
    plot_predictions(preds, labels)

# 运行主函数
main()
