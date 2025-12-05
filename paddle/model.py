import paddle
import paddle.nn as nn

class SimpleCNN(nn.Layer):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(input_dim, hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def forward(self, x):
        return self.classifier((self.base(x)))

class FE(nn.Layer):
    def __init__(self, input_dim, hidden_dims):
        super(FE, self).__init__()
        # PyTorch Conv2d(in, out, kernel) -> Paddle Conv2D(in, out, kernel)
        self.conv1 = nn.Conv2D(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(6, 16, 5)
        # PyTorch Linear(in, out) -> Paddle Linear(in, out)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        # view -> reshape
        x = paddle.reshape(x, [-1, 16 * 5 * 5])
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
class Classifier(nn.Layer):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(hidden_dims, output_dim)
    
    def forward(self, x):
        x = self.fc3(x)
        return x
    
def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)