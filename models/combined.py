import torch
from torch import nn
from .resnet import PreActResNet_cifar
import torch.nn.functional as F
import copy


class CombinedResNet(nn.Module):
    def __init__(self, source_model: PreActResNet_cifar, target_model: PreActResNet_cifar):
        super(CombinedResNet, self).__init__()
        self.source_model: PreActResNet_cifar = source_model
        self.freeze_source_model()
        self.target_model: PreActResNet_cifar = target_model
        self.alfa_source = self.get_alfa_empty_tensor(self.source_model, -0.5)
        self.alfa_target = self.get_alfa_empty_tensor(self.target_model, 0.5)

        self.combined_network = copy.deepcopy(self.source_model)
        self.combined_network.bn_last = nn.BatchNorm2d(self.source_model.bn_last.num_features)
        # num_classes jako paramter
        num_classes = 100
        self.combined_network.last = nn.Linear(self.source_model.bn_last.num_features, num_classes)
        #todo detach freezed parameters

    def freeze_source_model(self):
        for param in self.source_model.named_parameters():
            param[1].detach()

    def get_alfa_empty_tensor(self, model, value):
        alfas = []
        base = 32
        for param in model.named_parameters():
            # if 'conv1' in param[0] and param[1].shape[0] == param[1].shape[1] * 2 and param[1].shape[1] != 16:
            #     base //= 2
            if 'bn1' in param[0] and 'weight' in param[0] or 'conv1' in param[0] or 'conv2' in param[0]:
                layer = param[1]
                size_of_layer = layer.shape[0]
                alfa = torch.ones([size_of_layer, 1, 1]) * value
                alfas.append(alfa)
        return alfas

    def forward(self, x):
        alfa_idx = 0
        out = self.combine_results(self.source_model.conv1(x), self.target_model.conv1(x), alfa_idx)
        alfa_idx += 1
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
        out = {'All': self.logits(out.view(out.size(0), -1)) }
        return out

    def logits(self, x):
        x = self.combined_network.last(x)
        return x

    def combine_results(self, source, target, alfa_idx):
        alfa_source = self.alfa_source[alfa_idx]
        alfa_target = self.alfa_target[alfa_idx]
        # batch norm
        if len(source.shape) == 1:
            alfa_source_trimmed = alfa_source.view(alfa_source.size(0))
            alfa_target_trimmed = alfa_target.view(alfa_target.size(0))
            return (1 + alfa_source_trimmed) * source + alfa_target_trimmed * target
        # conv
        else:
            size_one = source.shape[-1]
            size_two = source.shape[-2]
            alfa_source_expanded = alfa_source.expand(-1, size_one, size_two)
            alfa_target_expanded = alfa_target.expand(-1, size_one, size_two)
            return (1 + alfa_source_expanded) * source + alfa_target_expanded * target


    def forward_stage(self, stage_source, stage_target, out, alfa_idx):
        for i in range(len(stage_source)):
            source_layer = stage_source[i]
            target_layer = stage_target[i]
            out = self.combine_layers(source_layer, target_layer, out, alfa_idx)
            alfa_idx += 3
        return out

    def combine_layers(self, source_layer, target_layer, out, alfa_idx):
        out = self.combine_results(source_layer.bn1(out), target_layer.bn1(out), alfa_idx)
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
        alfa_idx += 1
        self.fuze_stage(self.combined_network.stage1, self.source_model.stage1, self.target_model.stage1, alfa_idx)
        alfa_idx += 12
        self.fuze_stage(self.combined_network.stage2, self.source_model.stage2, self.target_model.stage2, alfa_idx)
        alfa_idx += 12
        self.fuze_stage(self.combined_network.stage3, self.source_model.stage3, self.target_model.stage3, alfa_idx)
        alfa_idx += 12
        self.fuze_stage(self.combined_network.stage4, self.source_model.stage4, self.target_model.stage4, alfa_idx)

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

        self.fuse_conv_bn_layer(combined_layer, source_layer, target_layer, alfa_idx)

        alfa_idx += 1
        combined_layer.conv2.weight =nn.Parameter( self.combine_results(source_layer.conv2.weight, target_layer.conv2.weight, alfa_idx))

    def fuse_bn(self, bn_combined, bn_source, bn_target, alfa_idx):
        bn_combined.weight = nn.Parameter(self.combine_results(bn_source.weight, bn_target.weight, alfa_idx))
        bn_combined.bias = nn.Parameter(self.combine_results(bn_source.bias, bn_target.bias, alfa_idx))
        bn_combined.weight.detach()
        bn_combined.bias.detach()

    def fuse_conv_bn_layer(self, combined_layer, source_layer, target_layer, alfa_idx):
        new_source_conv = self.fuse(source_layer.conv1, source_layer.bn2)
        new_target_conv = self.fuse(target_layer.conv1, target_layer.bn2)
        combined_layer.conv1.weight = nn.Parameter(self.combine_results(new_source_conv.weight, new_target_conv.weight, alfa_idx))

    def fuse(self, conv, bn):
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
