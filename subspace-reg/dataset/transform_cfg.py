from __future__ import print_function

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]
normalize = transforms.Normalize(mean=mean, std=std)


transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize([224,224]),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize([224,224]),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ])
]

transform_A_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize([224,224]),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.Resize([224,224]),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ])
]

# CIFAR style transformation
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean, std=std)
transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]

transform_D_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]


transform_M = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ])
]

transform_M_test = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        lambda x: np.array(x),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize
    ])
]

transforms_list = ['A', 'D']


transforms_options = {
    'A': transform_A,
    'D': transform_D,
    'M': transform_M,
}

transforms_test_options = {
    'A': transform_A_test,
    'D': transform_D_test,
    'M': transform_M_test,
}
