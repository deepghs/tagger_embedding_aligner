import numpy as np
import torch
from imgutils.tagging import convert_wd14_emb_to_prediction
from imgutils.tagging.wd14 import _get_wd14_weights
from torch import nn


class SuffixLinear(nn.Module):
    def __init__(self, initial_weights, initial_bias):
        nn.Module.__init__(self)

        if not isinstance(initial_weights, torch.Tensor):
            initial_weights = torch.tensor(initial_weights)
        self.register_buffer('weights', initial_weights)
        self.weights: torch.Tensor

        if not isinstance(initial_bias, torch.Tensor):
            initial_bias = torch.tensor(initial_bias)
        self.register_buffer('bias', initial_bias)
        self.bias: torch.Tensor

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(x @ self.weights + self.bias)


def get_suffix_linear(model_name: str = 'SwinV2_v3'):
    wx = _get_wd14_weights(model_name)
    return SuffixLinear(
        initial_weights=wx['weights'],
        initial_bias=wx['bias'],
    )


if __name__ == '__main__':
    model = get_suffix_linear()
    print(model)

    input_ = torch.randn(1024)
    with torch.no_grad():
        output = model(input_)
        print(output.shape)

    np_output = output.numpy()
    print(np_output.shape)

    pred = convert_wd14_emb_to_prediction(input_.numpy(), fmt='prediction')
    print(pred.shape)

    print(np.allclose(np_output, pred))
