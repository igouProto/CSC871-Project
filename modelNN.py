import torch.nn as nn

# class DigitClassifier(nn.Module):
#     def __init__(self, num_features, num_classes):
#         super().__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(num_features, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True),
#             nn.Linear(64, num_classes),
#             # nn.Softmax(dim=1)
#         )
    

#     def forward(self, x):
#         # Flatten the input image
#         x = x.view(-1, 28*28)
#         x = self.classifier(x)    
#         return x
class DigitClassifier(nn.Module):
    def __init__(self, num_features, num_classes, dropout_prob=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.Sigmoid(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes),
        )     

    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 28*28)
        x = self.classifier(x)    
        return x