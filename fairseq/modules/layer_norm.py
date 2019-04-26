# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn

class LayerNormImpl(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = ((x - mean) * (x - mean)).mean(-1, keepdim=True).sqrt()
        y = (x - mean) / (std + self.eps)

        if self.affine:
            y = self.weight * y + self.bias
        return y

def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=True):
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass

    if export:
        return LayerNormImpl(normalized_shape, eps, elementwise_affine)
        
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
