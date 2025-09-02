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


class GCNClassifier():
    def __init__(self, gcn_type, rks, pN, number_neighbors=40, graph_type="knn"):
        # Parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        self.pK = number_neighbors
        self.pN = pN
        self.rks = rks
        self.pLR = 0.0001
        self.pNNeurons = 32
        self.pNEpochs = 200
        self.gcn_type = gcn_type
        self.graph_type = graph_type
        #self.dynamic_Ks = self.compute_dynamic_ks(rks)
        #print(self.dynamic_Ks)
        #exit(1)

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
        #rk_file_path = "rk_flowers_ge_resnet_noemb.txt"
        #rks  = self.read_ranked_lists_file(self.top_k, rk_file_path)
        #tree = BallTree(self.x)
        #_, rks = tree.query(self.x, k=self.top_k)
        # compute traditional knn graph
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
