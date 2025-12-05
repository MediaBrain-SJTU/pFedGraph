import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleCNN_HyperNetwork(nn.Module):
    def __init__(self, n_nodes, embedding_dim, out_dim=10, hidden_dim=100, n_hidden_layer=3):
        super(SimpleCNN_HyperNetwork, self).__init__()
        self.out_dim = out_dim
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [nn.Linear(embedding_dim, hidden_dim)]
        for _ in range(n_hidden_layer):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(*layers)

        self.c1_weights = nn.Linear(hidden_dim, 6 * 3 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 6)
        self.c2_weights = nn.Linear(hidden_dim, 16 * 6 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 16)
        self.l1_weights = nn.Linear(hidden_dim, 120 * 16 * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 120)
        self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
        self.l2_bias = nn.Linear(hidden_dim, 84)
        self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
        self.l3_bias = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            "conv1.weight": self.c1_weights(features).view(6, 3, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(16, 6, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(120, 16 * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
            "fc2.weight": self.l2_weights(features).view(84, 120),
            "fc2.bias": self.l2_bias(features).view(-1),
            "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
            "fc3.bias": self.l3_bias(features).view(-1),
        })
        return weights

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)

def simplecnn_hypernetwork(num_nodes, embed_dim, hidden_dim, n_hidden_layer, out_dim):
    return SimpleCNN_HyperNetwork(num_nodes, embed_dim, out_dim=out_dim, hidden_dim=hidden_dim, n_hidden_layer=n_hidden_layer)