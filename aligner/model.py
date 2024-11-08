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


def get_model(model_name: str, n: int = 1024, suffix_model_name: str = 'SwinV2_v3', **options):
    converter = _KNOWN_CONVERTERS[model_name](n=n, **options)
    suffix = get_suffix_linear(suffix_model_name)
    return nn.Sequential(
        converter,
        suffix,
    )


if __name__ == '__main__':
    model = get_model('simple', n=1024)
    input_ = torch.randn(2, 1024)
    with torch.no_grad():
        output = model(input_)
        print(output.shape)
