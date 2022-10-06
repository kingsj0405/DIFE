import os
import torch
import torch.nn.functional as F
from .hourglass import (
    HourglassNet,
    ResidualBottleneckPreactivation,
)


class Transformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, domain_num=2):
        super().__init__()
        self.domain_num = domain_num
        for i in range(self.domain_num):
            self.add_module(f'interspecies2d{i}_layer1', torch.nn.Conv2d(input_dim, 64, kernel_size=7, padding=3))
            self.add_module(f'interspecies2d{i}_layer1_norm', torch.nn.GroupNorm(16, 64))
            self.add_module(f'interspecies2d{i}_layer2', torch.nn.Conv2d(64, 64, kernel_size=7, padding=3))
            self.add_module(f'interspecies2d{i}_layer2_norm', torch.nn.GroupNorm(16, 64))
            self.add_module(f'interspecies2d{i}_layer3', torch.nn.Conv2d(64, output_dim, kernel_size=1, padding=0))


    def transform(self, x, d1, d2):
        out = self._modules[f'{d1}2{d2}_layer1'](x)
        out = self._modules[f'{d1}2{d2}_layer1_norm'](out)
        out = F.relu(out)
        out = self._modules[f'{d1}2{d2}_layer2'](out)
        out = self._modules[f'{d1}2{d2}_layer2_norm'](out)
        out = F.relu(out)
        out = self._modules[f'{d1}2{d2}_layer3'](out)
        return out


class DIFE(torch.nn.Module):
    def __init__(
        self,
        dife_dim,
        domain_dim,
        domain_num,
    ):
        super().__init__()
        self.hg = HourglassNet(
            ResidualBottleneckPreactivation,
            use_group_norm=True,
            num_stacks=1,
            num_output_channels=dife_dim,
            output_as_tensor=True,
            keep_size=True,
        )
        self.transformer = Transformer(
            input_dim=dife_dim,
            output_dim=domain_dim,
            domain_num=domain_num,
        )
    
    def forward_hg(self, x):
        out = self.hg(x)
        return out
    
    def forward_transformer(self, x, d1, d2):
        out = self.transformer.transform(x, d1, d2)
        return out
    
    def forward(self, x, domain='interspecies'):
        if domain == 'interspecies':
            out = self.forward_hg(x)
        else:
            out = self.forward_transformer(x, 'interspecies', domain)
        return out

