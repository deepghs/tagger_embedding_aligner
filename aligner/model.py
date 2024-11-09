import os
from typing import Type, Optional, List, Union

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


class SimpleNumberBasedN2(nn.Module):
    def __init__(self, n: int = 1024, **kwargs):
        nn.Module.__init__(self)
        _ = kwargs

        self.fc1 = nn.Linear(n, int(n * 0.7))
        self.fc2 = nn.Linear(int(n * 0.7), 1)

    def forward(self, x):
        origin_x = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x * origin_x


register_converter('simple_num_n2', SimpleNumberBasedN2)


class NumMLP(nn.Module):
    def __init__(self, n: int = 1024, layers: Optional[List[Union[int, float]]] = None,
                 dropout_rate: float = 0.2, **kwargs):
        nn.Module.__init__(self)
        _ = kwargs
        layers = [
            x if isinstance(x, int) else int(round(x * n))
            for x in (layers or [])
        ]
        _layers = []

        layers = [n, *layers]
        for prev_layer, next_layer in zip(layers[:-1], layers[1:]):
            _layers.append(nn.Linear(prev_layer, next_layer))
            _layers.append(nn.ReLU())
            _layers.append(nn.Dropout(p=dropout_rate))
        _layers.append(nn.Linear(layers[-1], 1))
        self._layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self._layers(x)


_DEFAULT_LAYERS = [1.5, 0.5]


class NumBasedMLPConverter(nn.Module):
    def __init__(self, n: int = 1024, layers: Optional[List[Union[int, float]]] = None,
                 dropout_rate: float = 0.2, **kwargs):
        nn.Module.__init__(self)
        self.mlp = NumMLP(
            n=n,
            layers=_DEFAULT_LAYERS if layers is None else layers,
            dropout_rate=dropout_rate,
            **kwargs
        )

    def forward(self, x):
        rate = self.mlp(x)
        return x * rate


register_converter('num', NumBasedMLPConverter)


class Num0Converter(NumBasedMLPConverter):
    def __init__(self, n: int = 1024, **kwargs):
        NumBasedMLPConverter.__init__(
            self,
            n=n,
            layers=[],
            **kwargs
        )


class Num1Converter(NumBasedMLPConverter):
    def __init__(self, n: int = 1024, dropout_rate: float = 0.3, **kwargs):
        NumBasedMLPConverter.__init__(
            self,
            n=n,
            layers=[1.2],
            dropout_rate=dropout_rate,
            **kwargs,
        )


class Num2Converter(NumBasedMLPConverter):
    def __init__(self, n: int = 1024, dropout_rate: float = 0.3, **kwargs):
        NumBasedMLPConverter.__init__(
            self,
            n=n,
            layers=[0.5, 0.35],
            dropout_rate=dropout_rate,
            **kwargs,
        )


register_converter('num0', Num0Converter)
register_converter('num1', Num1Converter)
register_converter('num2', Num2Converter)


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


def open_model_with_data(model_file: str):
    if os.path.exists(model_file):
        if os.path.isfile(model_file):
            data = torch.load(model_file, map_location='cpu')
            model = get_model(**data['model_options'])
            existing_keys = set(model.state_dict())
            state_dict = {key: value for key, value in data['state_dict'].items() if key in existing_keys}
            model.load_state_dict(state_dict)
            return model, data
        else:
            return open_model_with_data(os.path.join(model_file, 'best.pt'))
    else:
        raise FileNotFoundError(f'Model {model_file!r} not found.')


def open_model(model_file: str):
    model, data = open_model_with_data(model_file)
    return model


if __name__ == '__main__':
    model = get_model('num', n=1024, dropout_rate=0.5)
    print(model)
    input_ = torch.randn(2, 1024)
    with torch.no_grad():
        prediction, embedding = model(input_)
        print(embedding.shape, embedding.dtype)
        print(prediction.shape, prediction.dtype)
