import torch.nn as nn
import torch
import os
import json
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))

class FE(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class Classifier(nn.Module):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        x = self.fc3(x)
        return x
    
def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)

class TextCNN_FE(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TextCNN_FE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = 100,
                kernel_size = (size, emb_size)
            )
            for size in [3, 4, 5]
        ])
        self.relu = nn.ReLU()
        
    def forward(self, text):
        embeddings = self.embedding(text).unsqueeze(1)  # (batch_size, 1, word_pad_len, emb_size)
        conved = [self.relu(conv(embeddings)).squeeze(3) for conv in self.convs]  # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]
        pooled = [torch.nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]
        flattened = torch.cat(pooled, dim = 1)  # (batch size, n_kernels * len(kernel_sizes))
        return flattened
    
class TextCNN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size):
        super(TextCNN, self).__init__()
        self.base = TextCNN_FE(vocab_size, emb_size)
        self.classifier = Classifier(300, n_classes)
        
    def forward(self, x):
        return self.classifier((self.base(x)))
    
def textcnn(n_classes):
    with open(os.path.join("/GPFS/data/zhenyangni/moonfm/data/yahoo_answers_csv/sents", 'word_map.json'), 'r') as j:
        word_map = json.load(j)
        vocab_size = len(word_map)
    return TextCNN(n_classes, vocab_size, 256)

        