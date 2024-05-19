import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.nn.functional import interpolate
import torchvision.models as models

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedMobileNetV2, self).__init__()
        self.mobilenet = mobilenet_v2(weights=None)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.resize = Resize((224, 224))
        # Modify the classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.mobilenet(x)