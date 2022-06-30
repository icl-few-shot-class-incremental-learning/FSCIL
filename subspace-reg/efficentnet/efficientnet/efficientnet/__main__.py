import sys

import torch

from .efficientnet import efficientnet0


torch.manual_seed(0xcafe)

net = efficientnet0(num_classes=64)

N, C, H, W = 2, 3, 224, 224
image = torch.rand(N, C, H, W)

out = net(image)
print(net)
print(out.shape)
print("It works!", file=sys.stderr)
