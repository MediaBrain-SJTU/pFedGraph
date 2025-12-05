import mindspore.nn as nn
from mindspore import ops

class SimpleCNN(nn.Cell):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def construct(self, x):
        return self.classifier((self.base(x)))

class FE(nn.Cell):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(input_dim, hidden_dims[0])
        self.fc2 = nn.Dense(hidden_dims[0], hidden_dims[1])
        self.flatten = nn.Flatten()
        
    def construct(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class Classifier(nn.Cell):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Dense(hidden_dims, output_dim)
    
    def construct(self, x):
        x = self.fc3(x)
        return x
    
def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)


        
