from .resnet_language import resnet12, resnet18
from .efficientnet_language import efficientnet0
import collections

EfficientNetParam = collections.namedtuple("EfficientNetParam", [
    "width", "depth", "resolution", "dropout"])

EfficientNetParams = {
  "B0": EfficientNetParam(1.0, 1.0, 224, 0.2)}

model_pool = [
    'resnet12',
    'resnet18',
    'efficientnet0'
]

model_dict = {
    'resnet12': resnet12,
    'resnet18': resnet18,
    'efficientnet0' : efficientnet0
}
