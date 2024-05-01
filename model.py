import torch
import torch.nn as nn

# the model
class DigitClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        # input layer
    ##    self.all_layers.add_module('input', torch.nn.Linear(num_features, neurons_per_layer))
      ##  self.all_layers.add_module('input_activation', activation_function)

        # hidden layers
      ##  for i in range(numbers_of_layers):
      ##      self.all_layers.add_module(f'hidden_{i}', torch.nn.Linear(neurons_per_layer, neurons_per_layer))
      ##      self.all_layers.add_module(f'hidden_{i}_activation', activation_function)

        # output layer
     ##   self.all_layers.add_module('output', torch.nn.Linear(neurons_per_layer, num_classes))
    

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28*28)
        x = self.classifier(x)    
        return x