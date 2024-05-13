import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN2(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 1 input channel, 6 output channels, kernel size of 5
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6 input channels (from previous layer), 16 output channels, kernel size of 5
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 * 4 * 4 input features (from convolutional layers), 120 output features
        self.fc2 = nn.Linear(120, 84)  # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 10)  # 84 input features, 10 output features (for 10 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the feature maps for the fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleCNN2(num_classes=10)