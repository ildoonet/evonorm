# https://github.com/digantamisra98/EvoNorm/blob/master/evonorm2d.py

import torch
import torch.nn as nn

from FastAutoAugment.layers.swish import SwishImplementation


def instance_std(x, eps=1e-5):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    return torch.sqrt(var + eps)


def group_std(x, groups=32, eps=1e-5):
    N, C, H, W = x.size()

    my_group_num = groups
    if groups > C:
        my_group_num = min(C, groups)
    if C % my_group_num != 0:
        for new_group in [32, 16, 8, 4, 2]:
            if new_group < my_group_num and C % new_group == 0:
                my_group_num = new_group

    x = x.view((N, my_group_num, C // my_group_num, H, W))
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.sqrt(var + eps).view((N, C, H, W))


# class MultSwishImplementation(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, i, v):
#         result = i * torch.sigmoid(v * i)
#         ctx.save_for_backward(i, v)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         i, v = ctx.saved_tensors
#         sigmoid_i = torch.sigmoid(i)
#         return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i))), v


class Identity(nn.Module):
    def forward(self, x):
        return x


class EvoNorm2DS0(nn.Module):
    def __init__(self, channels, groups=32):
        super(EvoNorm2DS0, self).__init__()
        self.channels = channels
        self.groups = groups
        self.gamma = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        self.v = nn.Parameter(torch.ones(1, self.channels, 1, 1))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        # assert not torch.isnan(self.v).any()
        num = x * torch.sigmoid(self.v * x)
        # num = MultSwishImplementation.apply(x, self.v)
        # assert not torch.isnan(num).any()
        a = num / group_std(x, self.groups)
        # assert not torch.isnan(a).any()
        out = torch.addcmul(self.beta, 1, self.gamma, a)
        # assert not torch.isnan(out).any()
        return out

    @classmethod
    def patch_frelu(cls):
        import gorilla
        import torch.nn.functional as F
        settings = gorilla.Settings(allow_hit=True)

        patch = gorilla.Patch(F, 'relu', lambda x: x, settings=settings)
        gorilla.apply(patch)

        print('F.relu have been patched to identity fn.')

    @classmethod
    def convert(cls, module: nn.Module, groups: int = 32) -> nn.Module:
        module_output = module

        if isinstance(module, nn.BatchNorm2d):
            module_output = EvoNorm2DS0(module.num_features, groups)
        elif isinstance(module, (nn.ReLU, )) or 'Swish' in module.__class__.__name__:
            # EvoNorm already includes Activation fn.
            module_output = Identity()

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert(child, groups))
        del module

        return module_output


class EvoNorm2DS1(nn.Module):
    def __init__(self, channels, groups=32):
        super(EvoNorm2DS1, self).__init__()
        self.channels = channels
        self.groups = groups
        self.gamma = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        num = SwishImplementation.apply(x)
        a = num / (group_std(x, self.groups) + 1e-8)
        # assert not torch.isnan(a).any()
        out = torch.addcmul(self.beta, 1, self.gamma, a)
        # assert not torch.isnan(out).any()
        return out

    @classmethod
    def patch_frelu(cls):
        import gorilla
        import torch.nn.functional as F
        settings = gorilla.Settings(allow_hit=True)

        patch = gorilla.Patch(F, 'relu', lambda x: x, settings=settings)
        gorilla.apply(patch)

        print('F.relu have been patched to identity fn.')

    @classmethod
    def convert(cls, module: nn.Module, groups: int = 32) -> nn.Module:
        module_output = module

        if isinstance(module, nn.BatchNorm2d):
            module_output = EvoNorm2DS1(module.num_features, groups)
        elif isinstance(module, (nn.ReLU, )) or 'Swish' in module.__class__.__name__:
            # EvoNorm already includes Activation fn.
            module_output = Identity()

        for name, child in module.named_children():
            module_output.add_module(name, cls.convert(child, groups))
        del module

        return module_output
