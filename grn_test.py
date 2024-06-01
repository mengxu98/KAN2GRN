import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from KAN.efficient_kan import KAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载单细胞表达谱数据
data = pd.read_csv(
    '../GRN/scGRN-L0_data/BEELINE-data/inputs/Synthetic/dyn-BF/dyn-BF-5000-10/ExpressionData.csv',
    header=0, index_col=0
    )
data = data.T

print("torch data")
expression_data = torch.tensor(data.values, dtype=torch.float32)

# 构建数据集和数据加载器
dataset = TensorDataset(expression_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# class KANLinear(torch.nn.Module):
#     def __init__(
#         self,
#         layers_hidden,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=torch.nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(KANLinear, self).__init__()
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         self.layers = torch.nn.ModuleList()
#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             self.layers.append(
#                 KANLinear(
#                     in_features,
#                     out_features,
#                     grid_size=grid_size,
#                     spline_order=spline_order,
#                     scale_noise=scale_noise,
#                     scale_base=scale_base,
#                     scale_spline=scale_spline,
#                     base_activation=base_activation,
#                     grid_eps=grid_eps,
#                     grid_range=grid_range,
#                 )
#             )

#     def forward(self, x: torch.Tensor, update_grid=False):
#         for layer in self.layers:
#             if update_grid:
#                 layer.update_grid(x)
#             x = layer(x)
#         return x

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         return sum(
#             layer.regularization_loss(regularize_activation, regularize_entropy)
#             for layer in self.layers
#         )

# 定义模型结构
input_dim = expression_data.shape[1]  # 基因数量
hidden_layers = [input_dim, 64, 32, input_dim]  # 隐藏层结构
model = KAN(layers_hidden=hidden_layers, grid_size=5, spline_order=3)

# model = KAN([51, 100, 4])
# model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss()  # 你可以根据需要选择适当的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        expression_batch = batch[0]
        
        # 前向传播
        outputs = model(expression_batch)
        loss = criterion(outputs, expression_batch)
        
        # 添加正则化损失
        reg_loss = model.regularization_loss()
        total_loss = loss + reg_loss
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')

# 提取最后一层的权重
last_layer = model.layers[-1]

# 提取基础权重和样条权重
base_weight = last_layer.base_weight.detach().cpu().numpy()
spline_weight = last_layer.spline_weight.detach().cpu().numpy()

# 如果有独立样条缩放器，也提取样条缩放器权重
if last_layer.enable_standalone_scale_spline:
    spline_scaler = last_layer.spline_scaler.detach().cpu().numpy()
    print("Spline Scaler Weights:\n", spline_scaler)

print("Base Weights:\n", base_weight)
print("Spline Weights:\n", spline_weight)

# 使用基础权重推断基因调控网络
# 根据基础权重的绝对值大小来构建基因调控网络
# 边的权重可以反映调控强度
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()


genes = data.columns
for i, gene_i in enumerate(genes):
    for j, gene_j in enumerate(genes):
        weight = base_weight[i, j]
        if abs(weight) > 0.1:  # 设定阈值来筛选重要的调控关系
            G.add_edge(gene_j, gene_i, weight=weight)

# 绘制基因调控网络
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()
