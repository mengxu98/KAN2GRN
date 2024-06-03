import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from KAN.efficient_kan import KAN
import networkx as nx

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load expression file
data = pd.read_csv(
    "../GRN/scGRN-L0_data/BEELINE-data/inputs/Synthetic/dyn-BF/dyn-BF-5000-10/ExpressionData.csv",
    header=0,
    index_col=0,
)
data = data.T

# Convert the data to a PyTorch tensor
print("Loading data...")
expression_data = torch.tensor(data.values, dtype=torch.float32)

dataset = TensorDataset(expression_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model structure
input_dim = expression_data.shape[1]  # Number of genes
# [51, 100, 4]
# hidden_layers = [input_dim, 64, 32, input_dim]  # Hidden layer structure
hidden_layers = [expression_data.shape[1], 100, expression_data.shape[1]]
model = KAN(layers_hidden=hidden_layers, grid_size=5, spline_order=3)
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # You can choose an appropriate loss function as needed
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        expression_batch = batch[0].to(device)  # Move batch to device

        # Forward pass
        outputs = model(expression_batch)
        loss = criterion(outputs, expression_batch)

        # Add regularization loss
        reg_loss = model.regularization_loss()
        total_loss = loss + reg_loss

        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}")

# Extract the weights of the last layer
last_layer = model.layers[-1]

# Extract base weights and spline weights
base_weight = last_layer.base_weight.detach().cpu().numpy()
spline_weight = last_layer.spline_weight.detach().cpu().numpy()

# If there is an independent spline scaler, also extract the spline scaler weights
if last_layer.enable_standalone_scale_spline:
    spline_scaler = last_layer.spline_scaler.detach().cpu().numpy()
    print("Spline Scaler Weights:\n", spline_scaler)

print("Base Weights:\n", base_weight)
print("Spline Weights:\n", spline_weight)

# Use base weights to infer gene regulatory network
# Construct the gene regulatory network based on the absolute values of the base weights
# Edge weights can reflect the strength of regulation

G = nx.DiGraph()

genes = data.columns
for i, gene_i in enumerate(genes):
    for j, gene_j in enumerate(genes):
        weight = base_weight[i, j]
        if (
            abs(weight) > 0.1
        ):  # Set a threshold to filter important regulatory relationships
            G.add_edge(gene_j, gene_i, weight=weight)

# Plot the gene regulatory network
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=700, font_size=10)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.show()
