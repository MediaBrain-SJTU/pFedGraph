import tensorflow as tf
from tensorflow.keras import layers, models, Model

class SimpleCNN(Model):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN, self).__init__()
        self.base = FE(hidden_dims)
        self.classifier = Classifier(hidden_dims[1], output_dim)

    def call(self, x):
        x = tf.convert_to_tensor(x)  # Ensure the input is a TensorFlow tensor
        x = self.base(x)
        return self.classifier(x)

class FE(Model):
    def __init__(self, hidden_dims):
        super(FE, self).__init__()
        # 显式为卷积层命名，保证不同实例之间变量名一致（便于按名字做聚合）
        self.conv1 = layers.Conv2D(6, 5, activation='relu', padding='valid', name='conv1')
        self.pool = layers.MaxPooling2D(2, 2)
        self.conv2 = layers.Conv2D(16, 5, activation='relu', padding='valid', name='conv2')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(hidden_dims[0], activation='relu', name='fc1')
        self.fc2 = layers.Dense(hidden_dims[1], activation='relu', name='fc2')

    def call(self, x):
        # Ensure the input shape is correct before passing to Conv2D layers
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
class Classifier(Model):
    def __init__(self, hidden_dims, output_dim=10):
        super(Classifier, self).__init__()
        self.fc3 = layers.Dense(output_dim, name='fc3')
    
    def call(self, x):
        x = self.fc3(x)
        return x

def simplecnn(n_classes):
    return SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=n_classes)
