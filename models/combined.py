import torch
from torch import nn
from .resnet import PreActResNet_cifar
import torch.nn.functional as F
import copy


class CombinedResNet(nn.Module):
    def __init__(self, source_model: PreActResNet_cifar, target_model: PreActResNet_cifar, num_classes: int, gpu: bool):
        super(CombinedResNet, self).__init__()
        self.gpu = gpu
        self.source_model: PreActResNet_cifar = source_model
        self.freeze_model(self.source_model)
        self.target_model: PreActResNet_cifar = target_model
        self.alfa_source = self.create_alfas(self.source_model, -0.5, 'source')
        self.alfa_target = self.create_alfas(self.target_model, 0.5, 'target')

        self.combined_network = copy.deepcopy(self.target_model)
        self.freeze_model(self.combined_network)
        self.combined_network.last['All'] = nn.Linear(self.target_model.bn_last.num_features, num_classes)

    def freeze_model(self, model):
        for param in model.parameters():
            param.detach_()

    def create_alfas(self, model, value, model_name):
        alfas = nn.ParameterDict()
        for param in model.named_parameters():
            name = param[0]
            layer = param[1]
            if self.alfa_condition(name):
                size_of_layer = layer.shape[0]
                alfa = torch.nn.Parameter(torch.ones(size_of_layer) * value)
                last_dot_index = name.rfind('.')
                name = name[:last_dot_index]
                # param_name = ('alfa.' + model_name + '.' + name).replace('.', '-')
                # self.register_parameter(param_name, alfa)
                if self.gpu:
                    alfa.cuda()
                alfas[name.replace('.', '-')] = alfa
        return alfas

    def alfa_condition(self, name):
        return 'bn1.weight' in name or 'bn2.weight' in name or 'conv2' in name or (
                'conv1' in name and not 'stage' in name) or 'shortcut' in name or 'bn_last.weight' in name

    def forward(self, x):
        out = self.combine_output(self.source_model.conv1(x), self.target_model.conv1(x), 'conv1')

        out = self.forward_stage(self.source_model.stage1, self.target_model.stage1, out, 'stage1')

        out = self.forward_stage(self.source_model.stage2, self.target_model.stage2, out, 'stage2')

        out = self.forward_stage(self.source_model.stage3, self.target_model.stage3, out, 'stage3')

        out = F.relu(self.combine_output(self.source_model.bn_last(out), self.target_model.bn_last(out), 'bn_last'))
        out = F.avg_pool2d(out, 8)
        out = self.logits(out.view(out.size(0), -1))
        return out

    def combine_output(self, source_output, target_output, alfa_key):
        alfa_source = self.alfa_source[alfa_key] + 1
        alfa_target = self.alfa_target[alfa_key]
        return source_output * alfa_source[None, :, None, None] + target_output * alfa_target[None, :, None, None]

    def logits(self, x):
        outputs = {}
        for task, func in self.combined_network.last.items():
            outputs[task] = func(x)
        return outputs

    def forward_stage(self, stage_source, stage_target, out, stage_key):
        for i in range(len(stage_source)):
            source_layer = stage_source[i]
            target_layer = stage_target[i]
            alfa_key = '-'.join([stage_key, str(i)])
            out = self.forward_layers(source_layer, target_layer, out, alfa_key)
        return out

    def forward_layers(self, source_layer, target_layer, x, alfa_key):
        alfa_key_bn1 = '-'.join([alfa_key, 'bn1'])
        out = self.combine_output(source_layer.bn1(x), target_layer.bn1(x), alfa_key_bn1)

        alfa_key_shortcut = '-'.join([alfa_key, 'shortcut'])
        shortcut = self.forward_shortcut(out, source_layer.shortcut, target_layer.shortcut,
                                         alfa_key_shortcut) if hasattr(source_layer, 'shortcut') else x
        out = F.relu(out)

        alfa_key_bn2 = '-'.join([alfa_key, 'bn2'])
        out = self.combine_output(source_layer.bn2(source_layer.conv1(out)), target_layer.bn2(target_layer.conv1(out)),
                                  alfa_key_bn2)
        out = F.relu(out)

        alfa_key_conv2 = '-'.join([alfa_key, 'conv2'])
        out = self.combine_output(source_layer.conv2(out), target_layer.conv2(out), alfa_key_conv2)

        out += shortcut
        return out

    def forward_shortcut(self, out, source_shortcut, target_shortcut, alfa_key):
        for i in range(len(source_shortcut)):
            alfa_key_shortcut = '-'.join([alfa_key, str(i)])
            out = self.combine_output(source_shortcut[i](out), target_shortcut[i](out), alfa_key_shortcut)
        return out

    def get_combined_network(self):
        self.fuse_conv(self.combined_network.conv1, self.source_model.conv1, self.target_model.conv1, 'conv1')

        self.fuse_stage(self.combined_network.stage1, self.source_model.stage1, self.target_model.stage1, 'stage1')

        self.fuse_stage(self.combined_network.stage2, self.source_model.stage2, self.target_model.stage2, 'stage2')

        self.fuse_stage(self.combined_network.stage3, self.source_model.stage3, self.target_model.stage3, 'stage3')

        self.fuse_bn(self.combined_network.bn_last, self.source_model.bn_last, self.target_model.bn_last, 'bn_last')

        return self.combined_network

    def fuse_stage(self, stage_combined, stage_source, stage_target, stage_key):
        for i in range(len(stage_source)):
            source_layer = stage_source[i]
            target_layer = stage_target[i]
            combined_layer = stage_combined[i]
            alfa_key = '-'.join([stage_key, str(i)])
            self.fuse_layers(combined_layer, source_layer, target_layer, alfa_key)

    def fuse_layers(self, combined_layer, source_layer, target_layer, alfa_key):
        alfa_key_bn1 = '-'.join([alfa_key, 'bn1'])
        self.fuse_bn(combined_layer.bn1, source_layer.bn1, target_layer.bn1, alfa_key_bn1)

        alfa_key_shortcut = '-'.join([alfa_key, 'shortcut'])
        if alfa_key_shortcut in self.alfa_source.keys():
            self.fuse_shortcut(combined_layer.shortcut, source_layer.shortcut, target_layer.shortcut, alfa_key_shortcut)

        # with fuse function
        # alfa_key_conv_bn = '-'.join([alfa_key, 'bn2'])
        # self.fuse_conv_bn_layer(combined_layer, source_layer, target_layer, alfa_key_conv_bn)

        # without fuse function
        alfa_key_conv_bn = '-'.join([alfa_key, 'bn2'])
        self.fuse_conv(combined_layer.conv1, source_layer.conv1, target_layer.conv1, alfa_key_conv_bn)
        self.fuse_bn(combined_layer.bn2, source_layer.bn2, target_layer.bn2, alfa_key_conv_bn)

        alfa_key_conv2 = '-'.join([alfa_key, 'conv2'])
        self.fuse_conv(combined_layer.conv2, source_layer.conv2, target_layer.conv2, alfa_key_conv2)

    def fuse_bn(self, bn_combined, bn_source, bn_target, alfa_key):
        new_weight = self.combine_weights_bn(bn_source.weight, bn_target.weight, alfa_key)
        bn_combined.weight = nn.Parameter(new_weight)
        new_bias = self.combine_weights_bn(bn_source.bias, bn_target.bias, alfa_key)
        bn_combined.bias = nn.Parameter(new_bias)

    def combine_weights_bn(self, source_weight, target_weight, alfa_key):
        alfa_source = self.alfa_source[alfa_key] + 1
        alfa_target = self.alfa_target[alfa_key]
        result = source_weight * alfa_source + target_weight * alfa_target
        return result

    def fuse_shortcut(self, combined_shortcut, source_shortcut, target_shortcut, alfa_key):
        for i in range(len(combined_shortcut)):
            alfa_key_shortcut = '-'.join([alfa_key, str(i)])
            self.fuse_conv(combined_shortcut[i], source_shortcut[i], target_shortcut[i], alfa_key_shortcut)

    def fuse_conv(self, combined_conv, source_conv, target_conv, alfa_key):
        new_weight = self.combine_weights_conv(source_conv.weight, target_conv.weight, alfa_key)
        combined_conv.weight = nn.Parameter(new_weight)

    def combine_weights_conv(self, source_weight, target_weight, alfa_key):
        alfa_source = self.alfa_source[alfa_key] + 1
        alfa_target = self.alfa_target[alfa_key]
        result = source_weight * alfa_source[:, None, None, None] + target_weight * alfa_target[:, None, None, None]
        return result

    def fuse_conv_bn_layer(self, combined_layer, source_layer, target_layer, alfa_key):
        new_source_conv = self.fuse(source_layer.conv1, source_layer.bn2)
        new_target_conv = self.fuse(target_layer.conv1, target_layer.bn2)

        self.fuse_conv(combined_layer.conv1, new_source_conv, new_target_conv, alfa_key)
        combined_layer.bn2 = nn.BatchNorm2d(source_layer.num_features)

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
