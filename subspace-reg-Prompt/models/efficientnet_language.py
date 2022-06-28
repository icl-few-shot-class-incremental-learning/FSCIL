

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
import math
import collections
import os
import pickle
from models.util import get_embeds


class LinearMap(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearMap, self).__init__()
        self.map = nn.Linear(indim, outdim)
        
    def forward(self, x):
        return self.map(x)
        
class LangPuller(nn.Module):
    def __init__(self,opt, vocab_base, vocab_novel):
        super(LangPuller, self).__init__()
        self.mapping_model = None
        self.opt = opt
        self.vocab_base = vocab_base
        self.vocab_novel = vocab_novel
        self.temp = opt.temperature
        dim = opt.word_embed_size # TODO

        # Retrieve novel embeds
        embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
        self.novel_embeds = get_embeds(embed_pth, vocab_novel).float().cuda()

        # Retrieve base embeds
        if opt.use_synonyms:
            embed_pth = os.path.join(opt.word_embed_path,
                                     "{0}_dim{1}_base_synonyms.pickle".format(opt.dataset, dim)) # TOdo
            with open(embed_pth, "rb") as openfile:
                label_syn_embeds = pickle.load(openfile)
            base_embeds = []
            for base_label in vocab_base:
                base_embeds.append(label_syn_embeds[base_label])
        else:
            embed_pth = os.path.join(opt.word_embed_path,
                                     "{0}_dim{1}.pickle".format(opt.dataset, dim)) # TODO
            base_embeds = get_embeds(embed_pth, vocab_base)

        self.base_embeds = base_embeds.float().cuda()
        # This will be used to compute label attractors.
        self.softmax = nn.Softmax(dim=1)
        # If Glove, use the first 300 TODO
        if opt.glove:
            self.base_embeds = self.base_embeds[:,:300]
            self.novel_embeds = self.novel_embeds[:,:300]
            
    def update_novel_embeds(self, vocab_novel):
        # Retrieve novel embeds
        opt = self.opt
        dim = opt.word_embed_size
        embed_pth = os.path.join(opt.word_embed_path, "{0}_dim{1}.pickle".format(opt.dataset, dim))
        new_novel_embeds = get_embeds(embed_pth, vocab_novel).float().cuda()
        self.novel_embeds = new_novel_embeds
        if opt.glove: #todo
            self.novel_embeds = self.novel_embeds[:,:300] # First 300 of the saved embeddings are Glove.
#         self.novel_embeds = torch.cat((self.novel_embeds, new_novel_embeds), 0)

    def create_pulling_mapping(self, state_dict, base_weight_size=640):
        indim = self.novel_embeds.size(1)
        outdim = base_weight_size
        self.mapping_model = LinearMap(indim, outdim)
        self.mapping_model.load_state_dict(state_dict)
        self.mapping_model = self.mapping_model.cuda()
        

    def forward(self, base_weight, mask=False):
        if self.mapping_model is None:
            # Default way of computing pullers is thru sem. sub. reg.:
            scores = self.novel_embeds @ torch.transpose(self.base_embeds, 0, 1)
            if mask:
                scores.fill_diagonal_(-9999)
            scores = self.softmax(scores / self.temp)
            return scores @ base_weight # 5 x 640 for fine-tuning.
        else:
            # Linear Mapping:
            with torch.no_grad():
                inspired = self.mapping_model(self.novel_embeds)
            return inspired

    def loss1(self, pull, inspired, weights):
        return pull * torch.norm(inspired - weights)**2

    def get_projected_weight(self, base_weight, weights):
        tr = torch.transpose(base_weight, 0, 1)
        Q, R = torch.qr(tr, some=True) # Q is 640x60
        mut = weights @ Q # mut is 5 x 60
        mutnorm = mut / torch.norm(Q.T, dim=1).unsqueeze(0)
        return mutnorm @ Q.T



        
EfficientNetParam = collections.namedtuple("EfficientNetParam", [
    "width", "depth", "resolution", "dropout"])

EfficientNetParams = {
  "B0": EfficientNetParam(1.0, 1.0, 224, 0.2)}

def efficientnet0(pretrained=False, progress=False,  **kwargs):
    return EfficientNet(param=EfficientNetParams["B0"], **kwargs)

class EfficientNet(nn.Module):
    def __init__(self, param, num_classes=100, vocab=None, opt=None):
        if vocab is not None:
            assert opt is not None

        print('assertion is passed')
        super().__init__()
       
        # For the exact scaling technique we follow the official implementation as the paper does not tell us
        # https://github.com/tensorflow/tpu/blob/01574500090fa9c011cb8418c61d442286720211/models/official/efficientnet/efficientnet_model.py#L101-L125

        def scaled_depth(n):
            return int(math.ceil(n * param.depth))

        # Snap number of channels to multiple of 8 for optimized implementations
        def scaled_width(n):
            n = n * param.width
            m = max(8, int(n + 8 / 2) // 8 * 8)

            if m < 0.9 * n:
                m = m + 8

            return int(m)

        self.conv1 = nn.Conv2d(3, scaled_width(32), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(scaled_width(32))
        self.relu = nn.ReLU6(inplace=True)

        self.layer1 = self._make_layer(n=scaled_depth(1), expansion=1, cin=scaled_width(32), cout=scaled_width(16), kernel_size=3, stride=1)
        self.layer2 = self._make_layer(n=scaled_depth(2), expansion=6, cin=scaled_width(16), cout=scaled_width(24), kernel_size=3, stride=2)
        self.layer3 = self._make_layer(n=scaled_depth(2), expansion=6, cin=scaled_width(24), cout=scaled_width(40), kernel_size=5, stride=2)
        self.layer4 = self._make_layer(n=scaled_depth(3), expansion=6, cin=scaled_width(40), cout=scaled_width(80), kernel_size=3, stride=2)
        self.layer5 = self._make_layer(n=scaled_depth(3), expansion=6, cin=scaled_width(80), cout=scaled_width(112), kernel_size=5, stride=1)
        self.layer6 = self._make_layer(n=scaled_depth(4), expansion=6, cin=scaled_width(112), cout=scaled_width(192), kernel_size=5, stride=2)
        self.layer7 = self._make_layer(n=scaled_depth(1), expansion=6, cin=scaled_width(192), cout=scaled_width(320), kernel_size=3, stride=1)

        self.features = nn.Conv2d(scaled_width(320), scaled_width(1280), kernel_size=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(param.dropout, inplace=True)
        #self.fc = nn.Linear(scaled_width(640), num_classes)
        ###############classifer layer로 
        self.vocab = vocab
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(scaled_width(1280), self.num_classes, bias=opt.linear_bias)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.zeros_(False)

        # Zero BatchNorm weight at end of res-blocks: identity by default
        # See https://arxiv.org/abs/1812.01187 Section 3.1
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.linear[1].weight)


    def _make_layer(self, n, expansion, cin, cout, kernel_size=3, stride=1):
        layers = []

        for i in range(n):
            if i == 0:
                planes = cin
                expand = cin * expansion
                squeeze = cout
                stride = stride
            else:
                planes = cout
                expand = cout * expansion
                squeeze = cout
                stride = 1

            layers += [Bottleneck(planes, expand, squeeze, kernel_size=kernel_size, stride=stride)]

        return nn.Sequential(*layers)


    def forward(self, x, is_feat=False, get_alphas=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.features(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)

        if self.num_classes > 0:
            if self.vocab is not None:
                x = self.classifier(x, get_alphas=get_alphas)
            else: # linear classifier has no attribute get_alphas
                x = self.classifier(x)

        return x

    def _get_base_weights(self):
        base_weight = self.classifier.weight.detach().clone().requires_grad_(False)
        if self.classifier.bias is not None:
            base_bias = self.classifier.bias.detach().clone().requires_grad_(False)
            return base_weight, base_bias
        else:
            return base_weight, None

    def augment_base_classifier_(self,
                                 n,
                                 novel_weight=None,
                                 novel_bias=None):

        # Create classifier weights for novel classes.
        base_device = self.classifier.weight.device
        base_weight = self.classifier.weight.detach()
        if self.classifier.bias is not None:
            base_bias = self.classifier.bias.detach()
        else:
            base_bias = None

        if novel_weight is None:
            novel_classifier = nn.Linear(base_weight.size(1), n, bias=(base_bias is not None)) # TODO!!
            novel_weight     = novel_classifier.weight.detach()
            if base_bias is not None and novel_bias is None:
                novel_bias = novel_classifier.bias.detach()

        augmented_weight = torch.cat([base_weight, novel_weight.to(base_device)], 0)
        self.classifier.weight = nn.Parameter(augmented_weight, requires_grad=True)

        if base_bias is not None:
            augmented_bias = torch.cat([base_bias, novel_bias.to(base_device)])
            self.classifier.bias = nn.Parameter(augmented_bias, requires_grad=True)


    def regloss(self, lmbd, base_weight, base_bias=None):
        reg = lmbd * torch.norm(self.classifier.weight[:base_weight.size(0),:] - base_weight)
        if base_bias is not None:
            reg += lmbd * torch.norm(self.classifier.bias[:base_weight.size(0)] - base_bias)**2
        return reg
    
    def reglossnovel(self, lmbd, novel_weight, novel_bias=None):
        rng1, rng2 = self.num_classes, self.num_classes + novel_weight.size(0)
        reg = lmbd * torch.norm(self.classifier.weight[rng1:rng2, :] - novel_weight) #**2??
        if novel_bias is not None:
            reg += lmbd * torch.norm(self.classifier.bias[rng1:rng2, :] - novel_bias)**2
        return reg




class Bottleneck(nn.Module):
    def __init__(self, planes, expand, squeeze, kernel_size, stride):
        super().__init__()

        self.expand = nn.Identity() if planes == expand else nn.Sequential(
            nn.Conv2d(planes, expand, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True))

        self.depthwise = nn.Sequential(
            nn.Conv2d(expand, expand, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=expand, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True))

        #self.scse = scSE(expand, r=0.25)

        self.linear = nn.Sequential(
            nn.Conv2d(expand, squeeze, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze))

        # Make all blocks skip-able via AvgPool + 1x1 Conv
        # See https://arxiv.org/abs/1812.01187 Figure 2 c

        downsample = []

        if stride != 1:
            downsample += [nn.AvgPool2d(kernel_size=stride, stride=stride)]

        if planes != squeeze:
            downsample += [
                nn.Conv2d(planes, squeeze, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(squeeze)]

        self.downsample = nn.Identity() if not downsample else nn.Sequential(*downsample)


    def forward(self, x):
        xx = self.expand(x)
        xx = self.depthwise(xx)
        #xx = self.scse(xx)
        xx = self.linear(xx)

        x = self.downsample(x)
        xx.add_(x)

        return xx


class cSE(nn.Module):
    def __init__(self, planes, r):
        super().__init__()

        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes, int(planes * r), kernel_size=1, bias=True),
            nn.ReLU6(inplace=True))

        self.expand = nn.Sequential(
            nn.Conv2d(int(planes * r), planes, kernel_size=1, bias=True),
            nn.Sigmoid())


    def forward(self, x):
        xx = self.squeeze(x)
        xx = self.expand(xx)

        return x * xx


class sSE(nn.Module):
    def __init__(self, planes):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(planes, 1, kernel_size=1, bias=True),
            nn.Sigmoid())


    def forward(self, x):
        xx = self.block(x)

        return x * xx


class scSE(nn.Module):
    def __init__(self, planes, r=0.25):
        super().__init__()

        self.cse = cSE(planes=planes, r=r)
        self.sse = sSE(planes=planes)


    def forward(self, x):
        return self.cse(x) + self.sse(x)


def swish(x, inplace=False):
    return  x * torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def hardsigmoid(x):
    return nn.functional.relu6(x + 3) / 6


class Hardsigmoid(nn.Module):
    def forward(self, x):
        return hardsigmoid(x)


def hardswish(x):
    return x * hardsigmoid(x)


class Hardswish(nn.Module):
    def forward(self, x):
        return hardswish(x)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, choices=['efficientnet0'])
    args = parser.parse_args()

    model_dict = {
        'efficientnet0': efficientnet0,
    }

    model = model_dict[args.model](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=64)
    data = torch.randn(2, 3, 84, 84)
    model = model.cuda()
    data = data.cuda()
    feat, logit = model(data, is_feat=True)
    #print('model_feat',feat[-1].shape)
    print('model_logit',logit.shape)
