import torch
from torch import nn
from .resnet import PreActResNet_cifar
import torch.nn.functional as F
import copy


class CombinedResNet(nn.Module):
    def __init__(self):
        self.alfa_source = None
        self.alfa_target = None

    def __init__(self, source_model: PreActResNet_cifar, target_model: PreActResNet_cifar, num_classes: int, gpu: bool):
        super(CombinedResNet, self).__init__()
        self.gpu = gpu
        self.source_model: PreActResNet_cifar = source_model
        self.freeze_model(self.source_model)
        self.target_model: PreActResNet_cifar = target_model
        self.alfa_source = self.get_alfa_empty_tensor(self.source_model, -0.5)
        self.alfa_target = self.get_alfa_empty_tensor(self.target_model, 0.5)

        self.combined_network = copy.deepcopy(self.target_model)
        # całkowity inny rozmiar conv1 żeby dopasować alfy
        self.combined_network.conv1 = nn.Conv2d(self.target_model.conv1.in_channels,
                                                self.target_model.conv1.out_channels,
                                                kernel_size=self.target_model.conv1.kernel_size,
                                                stride=self.target_model.conv1.stride,
                                                padding=self.target_model.conv1.padding)

        # nie ma nic w projekcie o kombinowaniu tego
        self.combined_network.bn_last = nn.BatchNorm2d(self.target_model.bn_last.num_features)
        self.combined_network.last['All'] = nn.Linear(self.target_model.bn_last.num_features, num_classes)

    def freeze_model(self, model):
        for param in model.named_parameters():
            layer = param[1]
            layer.detach()

    def get_alfa_empty_tensor(self, model, value):
        alfas = []
        for param in model.named_parameters():
            name = param[0]
            layer = param[1]
            if 'bn1' in name and 'weight' in name or 'conv1' in name and 'stage' in name or 'conv2' in name:
                size_of_layer = layer.shape[0]
                alfa = torch.ones([size_of_layer, 1, 1]) * value
                if self.gpu:
                    alfa = alfa.cuda()
                alfas.append(alfa)
        return alfas

    def forward(self, x):
        alfa_idx = 0
        # out = self.combine_results(self.source_model.conv1(x), self.target_model.conv1(x), alfa_idx)
        out = self.combined_network.conv1(x)

        out = self.forward_stage(self.source_model.stage1, self.target_model.stage1, out, alfa_idx)
        alfa_idx += 12
        out = self.forward_stage(self.source_model.stage2, self.target_model.stage2, out, alfa_idx)
        alfa_idx += 12
        out = self.forward_stage(self.source_model.stage3, self.target_model.stage3, out, alfa_idx)
        alfa_idx += 12
        # for sure that bn_last should be created in costructor?
        out = F.relu(self.combined_network.bn_last(out))
        out = F.avg_pool2d(out, 8)
        # and linear?
        out = self.logits(out.view(out.size(0), -1))
        return out

    def logits(self, x):
        outputs = {}
        for task, func in self.combined_network.last.items():
            outputs[task] = func(x)
        return outputs

    def combine_results(self, source, target, alfa_idx):
        alfa_source = self.alfa_source[alfa_idx]
        alfa_target = self.alfa_target[alfa_idx]
        # batch norm
        if len(source.shape) == 1:
            alfa_source_corrected = alfa_source.view(alfa_source.size(0))
            alfa_target_corrected = alfa_target.view(alfa_target.size(0))
        # conv
        else:
            size_one = source.shape[-1]
            size_two = source.shape[-2]
            alfa_source_corrected = alfa_source.expand(-1, size_one, size_two)
            alfa_target_corrected = alfa_target.expand(-1, size_one, size_two)

        return CombinedResNet.calculate_with_alfas(source, target, alfa_source_corrected, alfa_target_corrected)

    @staticmethod
    def calculate_with_alfas(source, target, alfa_source, alfa_target):
        return (1 + alfa_source) * source + alfa_target * target

    def forward_stage(self, stage_source, stage_target, out, alfa_idx):
        for i in range(len(stage_source)):
            source_layer = stage_source[i]
            target_layer = stage_target[i]
            out = self.combine_layers(source_layer, target_layer, out, alfa_idx)
            alfa_idx += 3
        return out

    def combine_layers(self, source_layer, target_layer, out, alfa_idx):
        out = self.combine_results(source_layer.bn1(out), target_layer.bn1(out), alfa_idx)
        # nie wiem co z tym shortcutem
        # shortcut = source_layer.shortcut(out)
        out = F.relu(out)
        out = self.combine_results(source_layer.bn2(source_layer.conv1(out)), target_layer.bn2(target_layer.conv1(out)), alfa_idx + 1)
        out = F.relu(out)
        out = self.combine_results(source_layer.conv2(out), target_layer.conv2(out), alfa_idx + 2)
        # out += shortcut
        return out

    def get_combined_network(self):
        alfa_idx = 0
        # new_conv1_weights = self.combine_results(self.source_model.conv1.weight, self.target_model.conv1.weight, alfa_idx)
        # self.combined_network.conv1.weight = new_conv1_weights

        self.fuze_stage(self.combined_network.stage1, self.source_model.stage1, self.target_model.stage1, alfa_idx)
        alfa_idx += 12
        self.fuze_stage(self.combined_network.stage2, self.source_model.stage2, self.target_model.stage2, alfa_idx)
        alfa_idx += 12
        self.fuze_stage(self.combined_network.stage3, self.source_model.stage3, self.target_model.stage3, alfa_idx)
        # alfa_idx += 12
        # self.fuze_stage(self.combined_network.stage4, self.source_model.stage4, self.target_model.stage4, alfa_idx)
        return self.combined_network

    def fuze_stage(self, stage_combined, stage_source, stage_target, alfa_idx):
        for i in range(len(stage_source)):
            source_layer = stage_source[i]
            target_layer = stage_target[i]
            combined_layer = stage_combined[i]
            self.fuze_layers(combined_layer, source_layer, target_layer, alfa_idx)
            alfa_idx += 3

    def fuze_layers(self, combined_layer, source_layer, target_layer, alfa_idx):
        self.fuse_bn(combined_layer.bn1, source_layer.bn1, target_layer.bn1, alfa_idx)
        alfa_idx += 1

        # różny rozmiar conv1 i bn2 ( 32, 16, 3, 3) do 32 ( 16 != 32 )
        # self.fuse_conv_bn_layer(combined_layer, source_layer, target_layer, alfa_idx)

        self.fuse_bn(combined_layer.bn2, source_layer.bn2, target_layer.bn2, alfa_idx)

        alfa_idx += 1
        new_weight = self.combine_results(source_layer.conv2.weight, target_layer.conv2.weight, alfa_idx)
        combined_layer.conv2.weight = nn.Parameter(new_weight)

    def fuse_bn(self, bn_combined, bn_source, bn_target, alfa_idx):
        new_weight = self.combine_results(bn_source.weight, bn_target.weight, alfa_idx)
        bn_combined.weight = nn.Parameter(new_weight)
        new_bias = self.combine_results(bn_source.bias, bn_target.bias, alfa_idx)
        bn_combined.bias = nn.Parameter(new_bias)

    def fuse_conv_bn_layer(self, combined_layer, source_layer, target_layer, alfa_idx):
        new_source_conv = self.fuse(source_layer.conv1, source_layer.bn2)
        new_target_conv = self.fuse(target_layer.conv1, target_layer.bn2)

        new_weight = self.combine_results(new_source_conv.weight, new_target_conv.weight, alfa_idx)
        combined_layer.conv1.weight = nn.Parameter(new_weight)
        combined_layer.bn2 = nn.BatchNorm2d(source_layer.num_features)
        combined_layer.bn2.weight.detach()
        combined_layer.bn2.bias.detach()

    @staticmethod
    def fuse(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        beta = bn.weight
        gamma = bn.bias
        if conv.bias is not None:
            b = conv.bias
        else:
            b = mean.new_zeros(mean.shape)
        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean) / var_sqrt * beta + gamma
        fused_conv = nn.Conv2d(conv.in_channels,
                               conv.out_channels,
                               conv.kernel_size,
                               conv.stride,
                               conv.padding,
                               bias=True)
        fused_conv.weight = nn.Parameter(w)
        fused_conv.bias = nn.Parameter(b)
        return fused_conv
