"""

░██████╗░░█████╗░███╗░░██╗  ████████╗██████╗░███████╗██╗███╗░░██╗░█████╗░███╗░░░███╗███████╗███╗░░██╗████████╗░█████╗░
██╔════╝░██╔══██╗████╗░██║  ╚══██╔══╝██╔══██╗██╔════╝██║████╗░██║██╔══██╗████╗░████║██╔════╝████╗░██║╚══██╔══╝██╔══██╗
██║░░██╗░██║░░╚═╝██╔██╗██║  ░░░██║░░░██████╔╝█████╗░░██║██╔██╗██║███████║██╔████╔██║█████╗░░██╔██╗██║░░░██║░░░██║░░██║
██║░░╚██╗██║░░██╗██║╚████║  ░░░██║░░░██╔══██╗██╔══╝░░██║██║╚████║██╔══██║██║╚██╔╝██║██╔══╝░░██║╚████║░░░██║░░░██║░░██║
╚██████╔╝╚█████╔╝██║░╚███║  ░░░██║░░░██║░░██║███████╗██║██║░╚███║██║░░██║██║░╚═╝░██║███████╗██║░╚███║░░░██║░░░╚█████╔╝
░╚═════╝░░╚════╝░╚═╝░░╚══╝  ░░░╚═╝░░░╚═╝░░╚═╝╚══════╝╚═╝╚═╝░░╚══╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░

███████╗  ██╗███╗░░██╗███████╗███████╗██████╗░███████╗███╗░░██╗░█████╗░██╗░█████╗░
██╔════╝  ██║████╗░██║██╔════╝██╔════╝██╔══██╗██╔════╝████╗░██║██╔══██╗██║██╔══██╗
█████╗░░  ██║██╔██╗██║█████╗░░█████╗░░██████╔╝█████╗░░██╔██╗██║██║░░╚═╝██║███████║
██╔══╝░░  ██║██║╚████║██╔══╝░░██╔══╝░░██╔══██╗██╔══╝░░██║╚████║██║░░██╗██║██╔══██║
███████╗  ██║██║░╚███║██║░░░░░███████╗██║░░██║███████╗██║░╚███║╚█████╔╝██║██║░░██║
╚══════╝  ╚═╝╚═╝░░╚══╝╚═╝░░░░░╚══════╝╚═╝░░╚═╝╚══════╝╚═╝░░╚══╝░╚════╝░╚═╝╚═╝░░╚═╝


// Ian Bezerra ICMC-USP//

"""
#---------------------------------------

import gc
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import BallTree
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SGConv
from torch_geometric.nn import APPNP
from torch_geometric.nn import ARMAConv


#---------------------------------------



# Definicao das Classes(nn.module) das GCNs

class ARMA(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(ARMA, self).__init__()

        self.conv1 = ARMAConv(num_features, 16, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)

        self.conv2 = ARMAConv(16, num_classes, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(APPNPNet, self).__init__()
        hidden = 64
        K = 10
        alpha = 0.1
        self.lin1 = Linear(num_features, hidden)
        self.lin2 = Linear(hidden, num_classes)
        self.prop1 = APPNP(K, alpha)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        dropout = 0.5
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, K=2, cached=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class Net(torch.nn.Module):
    def __init__(self, pNFeatures, pNNeurons, numberOfClasses):
        super(Net, self).__init__()
        self.conv1 = GCNConv(pNFeatures, pNNeurons) #dataset.num_node_features
        self.conv2 = GCNConv(pNNeurons, numberOfClasses) #dataset.num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


#------// Classe Geral //---------------------------------


class GCNClassifier():
    def __init__(self, gcn_type, rks, pN, k, pNNeurons, graph_type="knn"):
        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = 0.0001
        self.pNNeurons = pNNeurons
        self.pNEpochs = 200
        self.gcn_type = gcn_type
        self.graph_type = graph_type


    def fit(self, test_index, train_index, features, labels):
        # masks
        print('Creating masks ...')
        self.train_mask = []
        self.val_mask = []
        self.test_mask = []
        self.train_size = len(train_index)
        self.test_size = len(test_index)
        self.train_mask = [False for i in range(self.pN)]
        self.val_mask = [False for i in range(self.pN)]
        self.test_mask = [False for i in range(self.pN)]
        for index in train_index:
            self.train_mask[index] = True
        for index in test_index:
            self.test_mask[index] = True
        self.train_mask = torch.tensor(self.train_mask)
        self.val_mask = torch.tensor(self.val_mask)
        self.test_mask = torch.tensor(self.test_mask)
        # labels
        print('Set labels ...')
        y = labels
        self.numberOfClasses = max(y)+1
        self.y = torch.tensor(y).to(self.device)
        # features
        self.x = torch.tensor(features).to(self.device)
        self.pNFeatures = len(features[0])
        # build graph
        self.create_graph()

    def read_ranked_lists_file(self, top_k, file_path):
        print("\tReading file", file_path)
        with open(file_path, 'r') as f:
            return [[int(y) for y in x.strip().split(' ')][:top_k] for x in f.readlines()]

    def create_graph(self):
        print('Making edge list ...')
        self.top_k = self.pK
        if self.graph_type == "rec":
            refList = [[] for i in range(self.pN)]
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    refList[img2].append(img1)
            edge_index = []
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    if img2 in refList[img1]:
                        edge_index.append([img1, img2])
        elif self.graph_type == "knn":
            edge_index = []
            for img1 in range(len(self.rks)):
                for pos in range(self.top_k):
                    img2 = self.rks[img1][pos]
                    edge_index.append([img1, img2])
        edge_index = torch.tensor(edge_index)
        # convert to torch format
        self.edge_index = edge_index.t().contiguous().to(self.device)

    def predict(self):
        # data object
        print('Loading data object...')
        data = Data(x=self.x.float(),
                    edge_index=self.edge_index,
                    y=self.y,
                    test_mask=self.test_mask,
                    train_mask=self.train_mask,
                    val_mask=self.val_mask)
        # TRAIN MODEL #
        if self.gcn_type == "gcn_net":
            model = Net(self.pNFeatures, self.pNNeurons, self.numberOfClasses).to(self.device)
        elif self.gcn_type == "gcn_gat":
            model = GAT(self.pNFeatures, self.numberOfClasses).to(self.device)
        elif self.gcn_type == "gcn_sgc":
            model = SGC(self.pNFeatures, self.numberOfClasses).to(self.device)
        elif self.gcn_type == "gcn_appnpnet":
            model = APPNPNet(self.pNFeatures, self.numberOfClasses).to(self.device)
        elif self.gcn_type == "gcn_arma":
            model = ARMA(self.pNFeatures, self.numberOfClasses).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.pLR, weight_decay=5e-4)

        print('Training...')
        model.train()
        for epoch in range(self.pNEpochs):
            print("Training epoch: ", epoch)
            optimizer.zero_grad()
            out = model(data)
            data.y = torch.tensor(data.y, dtype=torch.long)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        # MODEL EVAL #
        model.eval()
        _, pred = model(data).max(dim=1)
        pred = torch.masked_select(pred, data.test_mask.to(self.device))

        return pred.tolist()


#--------------------------------------------------------------------
# Parte Para Implementacao All-In-One Para o Servidor!!

"""
 -> Recolhendo Material!!

 /Imgs - S3 e Servidor
 List.txt - Servidor
 Classes.txt - S3
 /Emb - Servidor
 /Runs(rks) - Servidor

 -> Nao usado para GCN
 
 /Plots - S3

"""




#----// Opcoes de GCN //--------------------------------
gcn_options = {
    1: "gcn_net",
    2: "gcn_gat",
    3: "gcn_sgc",
    4: "gcn_appnpnet",
    5: "gcn_arma",
}

#----// Selecionando o modelo de extrator //--------------------------------

from Lib import model_options

print("Selecione o modelo de extrator:")
for key, value in model_options.items():
    print(f"{key}: {value}")

extrator_id = int(input("Selecione o modelo de extrator:"))
extrator = model_options[extrator_id]


#----// Selecionando o modelo de GCN //--------------------------------

print("Selecione o modelo de GCN:")
for key, value in gcn_options.items():
    print(f"{key}: {value}")


gcn_index = int(input("Digite o número do modelo de GCN: "))
gcn_type = gcn_options[gcn_index]


#----// Preparando Labels!! //--------------------------------

with open("list.txt", "r") as f: # No servidor /Imgs
    images = [line.strip() for line in f if line.strip()]


# Dicionario para mapear os labels!!
label_dict = {}
with open("Classes.txt", "r") as lf:  # Pegar de S3
    for line in lf:
        line = line.strip()
        if not line or ":" not in line:
            continue
        img, label = line.split(":", 1)
        label_dict[img.strip()] = int(label.strip())


# Agora com o mesmo indice que o extrator!!
labels = [label_dict[img] for img in images]



#----// Recontruindo o emb //--------------------------------
# Load with numpy and convert to torch tensor
import numpy as np
features_np = np.load(f"{extrator}_emb.npy", allow_pickle=True)
features = torch.tensor(features_np, dtype=torch.float)   # No servidor

#import torch
#features = torch.load(f"{extrator}_emb.pt")


#train_indices = [...]  # Indices for training nodes (fiquei com duvidas...)
#test_indices = [...]   # Indices for test nodes (fiquei com duvidas...)




#----// Recontruindo o rks //--------------------------------

import os
import json
import numpy as np


rks_path = "" + extrator + "_output.json"   # No servidor

with open(rks_path, "r") as f:
    rankings = json.load(f)

# Convert to numpy array
rks = np.array(rankings)


#----// Inicializando o GCN //--------------------------------


pN = len(features)  

from Tools import run
import utils

n_executions = 5   # Get from FrontEnd
n_folds = 10     # Get from FrontEnd

# Split data in folds
folds = utils.fold_split(features, labels, n_folds=n_folds)

pNNeurons = 32 # Get From FrontEnd
    
#----// inferencia do GCN //--------------------------------

# def run(features, labels, folds, rks, gcn_type, pNNeurons, graph_type="knn"):

# Define number of neighbors (k) for the graph
number_neighbors = 10  # You can adjust this value as needed

# List to store results from all runs
all_results = []

print(f"\nRunning {n_executions} executions with {gcn_type}...\n")

# Run multiple executions and collect results
for i in range(n_executions):
    print(f"Execution {i+1}/{n_executions}")
    results = run(
        features,
        labels,
        folds,
        rks,
        gcn_type,
        pNNeurons,
        GCNClassifier,  # Pass the GCNClassifier class
        number_neighbors  # Pass k parameter
    )
    all_results.append(results)
    
    # Calculate and print accuracy for this execution
    execution_accuracies = [fold_result[2] for fold_result in results]  # Extract accuracies
    avg_accuracy = sum(execution_accuracies) / len(execution_accuracies)
    print(f"Extractor: {extrator} and GCN: {gcn_type}")
    print(f"Execution {i+1} average accuracy: {avg_accuracy:.4f}")

# Calculate overall average accuracy across all executions and folds
all_accuracies = []
for execution_results in all_results:
    for fold_result in execution_results:
        all_accuracies.append(fold_result[2])  # Extract accuracy value

overall_avg_accuracy = sum(all_accuracies) / len(all_accuracies)
print(f"\n{'-'*50}")
print(f"Overall average accuracy across {n_executions} executions and {n_folds} folds: {overall_avg_accuracy:.4f}")
print(f"{'-'*50}\n")

# Print detailed fold results in a table
print(f"{'='*80}")
print(f"{'Detailed Results by Fold':^80}")
print(f"{'='*80}")
print(f"{'Execution':<10}{'Fold':<10}{'Accuracy':<15}")
print(f"{'-'*80}")

# Create a table with all fold results
for exec_idx, execution_results in enumerate(all_results):
    for fold_idx, fold_result in enumerate(execution_results):
        accuracy = fold_result[2]  # Extract accuracy value
        print(f"{exec_idx+1:<10}{fold_idx+1:<10}{accuracy:.4f}{'':15}")
    # Add a separator between executions
    if exec_idx < len(all_results) - 1:
        print(f"{'-'*80}")

print(f"{'='*80}\n")



