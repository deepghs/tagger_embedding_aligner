from typing import Type

import torch
from torch import nn

from .suffix import get_suffix_linear

_KNOWN_CONVERTERS = {}


def register_converter(model_name: str, model_cls: Type[nn.Module]):
    _KNOWN_CONVERTERS[model_name] = model_cls


class SimpleEmbeddingConverter(nn.Module):
    def __init__(self, n: int = 1024, **kwargs):
        nn.Module.__init__(self)
        _ = kwargs
        self.ln = nn.LayerNorm(n)
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln(x)
        y1 = self.fc1(x)
        y2 = self.gelu(self.fc2(x))
        return y1 * y2


register_converter('simple', SimpleEmbeddingConverter)


class SimpleNumberBased(nn.Module):
    def __init__(self, n: int = 1024, **kwargs):
        nn.Module.__init__(self)
        _ = kwargs

        self.fc1 = nn.Linear(n, int(n * 1.5))
        self.fc2 = nn.Linear(int(n * 1.5), n // 2)
        self.fc3 = nn.Linear(n // 2, 1)

    def forward(self, x):
        origin_x = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x * origin_x


register_converter('simple_num', SimpleNumberBased)


class _Model(nn.Module):
    def __init__(self, converter: nn.Module, suffix: nn.Module):
        nn.Module.__init__(self)
        self.converter = converter
        self.suffix = suffix

    def forward(self, x):
        y = self.converter(x)
        return self.suffix(y), y


def get_model(model_name: str, n: int = 1024, suffix_model_name: str = 'SwinV2_v3', **options):
    converter = _KNOWN_CONVERTERS[model_name](n=n, **options)
    suffix = get_suffix_linear(suffix_model_name)
    return _Model(converter, suffix)


if __name__ == '__main__':
    model = get_model('simple', n=1024)
    input_ = torch.randn(2, 1024)
    with torch.no_grad():
        output = model(input_)
        print(output.shape)
