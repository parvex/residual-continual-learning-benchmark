from torch import nn
from resnet import PreActResNet_cifar


class CombinedResNet(nn.Module):

    def __init__(self, source_model: PreActResNet_cifar, target_model: PreActResNet_cifar):
        super(CombinedResNet, self).__init__()
        self.source_model = source_model
        self.target_model = target_model
