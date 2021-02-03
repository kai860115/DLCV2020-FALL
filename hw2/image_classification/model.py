import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        self.vgg.classifier[6] = nn.Linear(4096, 50)
    
    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = self.vgg.classifier[:-3](x)
        x = self.vgg.classifier(x)
        return feature, x
