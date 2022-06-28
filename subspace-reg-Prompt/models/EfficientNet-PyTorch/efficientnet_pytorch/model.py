"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).
import numpy
import torch
from torch import nn
from torch.nn import functional as F
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.parameter.Parameter(nn.init.uniform_(torch.FloatTensor(10,5,1280),0,0.01))
        self.prompt_key = nn.parameter.Parameter(nn.init.uniform_(torch.FloatTensor(10,1280),0,0.01))

    def l2_normalize(self, x, axis=None, epsilon=1e-12, size=10):
        epsilons = torch.cuda.FloatTensor([[epsilon]]*size)

        square_sum = torch.sum(torch.square(x), axis, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, epsilons))
        return x * x_inv_norm

    def expand_to_batch(self, x, batch_size):
        return torch.tile(torch.unsqueeze(x, 0), [batch_size] + [1 for _ in x.shape])
    
    def prefix_prompt(self, prompt, x):
        return torch.cat((prompt, x), dim=1)
        
    def forward(self, x):
        # x = bs*1000
        prompt_norm = self.l2_normalize(self.prompt_key, axis=1) # 10*1000
        x_embed_norm = self.l2_normalize(x, axis=1, size=1) # bs*1000
        
        similarity = torch.matmul(x_embed_norm, torch.transpose(prompt_norm, 0, 1)) # bs*10
        (_, idx) = torch.topk(similarity, 5)
        prompt_id, id_counts = torch.unique(idx, return_counts=True)
        _, major_idx = torch.topk(id_counts, 5)
        major_prompt_id = prompt_id[major_idx]
        major_prompt_id = torch.sort(major_prompt_id)[0]
        idx = self.expand_to_batch(major_prompt_id, x_embed_norm.shape[0])
        
        batched_prompt_raw = torch.index_select(self.prompt, 0, idx[0])
        batched_prompt_raw = self.expand_to_batch(batched_prompt_raw, x_embed_norm.shape[0])
        # print(batched_prompt_raw)
        # print(batched_prompt_raw.shape)
        
        bs, allowed_size, prompt_len, embed_dim = batched_prompt_raw.shape
        
        # 이 부분만 좀 다름
        batched_prompt = torch.reshape(batched_prompt_raw, (bs, allowed_size*prompt_len*embed_dim))
        
        res = dict()
        
        res["prompt_idx"] = idx
        res["prompt_norm"] = prompt_norm
        res["x_embed_norm"] = x_embed_norm
        res["sim"] = similarity
        
        batched_key_norm = torch.index_select(prompt_norm, 0, idx[0])
        batched_key_norm = self.expand_to_batch(batched_key_norm, x_embed_norm.shape[0])
        res["selected_key"] = batched_key_norm
        # print(x_embed_norm)
        # print(x_embed_norm.shape)
        x_embed_norm = x_embed_norm[:, numpy.newaxis, :]
        # print(x_embed_norm)
        # print(x_embed_norm.shape)
        
        sim = batched_key_norm * x_embed_norm
        # print(sim)
        # print(sim.shape)
        reduce_sim = torch.sum(sim) / x_embed_norm.shape[0]
        res["reduce_sim"] = reduce_sim
        
        res["prompted_embedding"] = self.prefix_prompt(batched_prompt, x)
        # print(res)
        
        return res

class EfficientNet(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:
        >>> import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        # 이거 두개
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            
            # print(out_channels)
            # print(self._global_params)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()
    
    def add_prompt(self):
        
        self.prompt_module = Prompt()
        
        # prompt 버전
        if self._global_params.include_top:
            self._fc = nn.Linear(33280, self._global_params.num_classes, False)

        
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints
    
    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        
        # Convolution layers
        
        # torch.Size([64, 3, 224, 224])
        # torch.Size([64, 1280, 7, 7])
        # torch.Size([64, 1280, 1, 1])
        # torch.Size([64, 1280])
        # torch.Size([64, 1280])
        # torch.Size([64, 100])
        # torch.Size([64, 100])
        
        
        # print(inputs.shape)
        x = self.extract_features(inputs)
        # print(x.shape)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        # print(x.shape)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            # 여기 우선
            res_prompt = self.prompt_module(x)
            prompted_x = res_prompt['prompted_embedding']
            
            # print(x.shape)
            # print(prompted_x.shape)
            x = self._dropout(prompted_x)
            # 또는 여기
            # print(x.shape)
            #! HSJ bias False
            #self._fc.bias = None
            x = self._fc(x)
            # print(x.shape)
            
            # print(x.shape)
            # x = self._dropout(x)
            # # 또는 여기
            # print(x.shape)
            # #! HSJ bias False
            # #self._fc.bias = None
            # x = self._fc(x)
            # print(x.shape)
            
        return x, res_prompt

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))
    
    
    def _get_base_weights(self):
        base_weight = self._fc.weight.detach().clone().requires_grad_(False)
        if self._fc.bias is not None:
            base_bias = self._fc.bias.detach().clone().requires_grad_(False)
            return base_weight, base_bias
        else:
            return base_weight, None

   
    def augment_base_classifier_(self,
                                 n,
                                 novel_weight=None,
                                 novel_bias=None):

        # Create classifier weights for novel classes.
        base_device = self._fc.weight.device
        base_weight = self._fc.weight.detach()
        if self._fc.bias is not None:
            base_bias = self._fc.bias.detach()
        else:
            base_bias = None

        if novel_weight is None:
            novel_classifier = nn.Linear(base_weight.size(1), n, bias=(base_bias is not None)) # TODO!!
            novel_weight = novel_classifier.weight.detach()
            if base_bias is not None and novel_bias is None:
                novel_bias = novel_classifier.bias.detach()

        augmented_weight = torch.cat([base_weight, novel_weight.to(base_device)], 0)
        self._fc.weight = nn.Parameter(augmented_weight, requires_grad=True)
        print(self._fc.weight.shape)

        if base_bias is not None:
            augmented_bias = torch.cat([base_bias, novel_bias.to(base_device)])
            self._fc.bias = nn.Parameter(augmented_bias, requires_grad=True)

    
    def regloss(self, lmbd, base_weight, base_bias=None):
        reg = lmbd * torch.norm(self._fc.weight[:base_weight.size(0),:] - base_weight)
        if base_bias is not None:
            reg += lmbd * torch.norm(self._fc.bias[:base_weight.size(0)] - base_bias)**2
        return reg
    
    
    def reglossnovel(self, lmbd, novel_weight, novel_bias=None):
        rng1, rng2 = self._global_params.num_classes, self._global_params.num_classes + novel_weight.size(0)
        reg = lmbd * torch.norm(self._fc.weight[rng1:rng2, :] - novel_weight) #**2??
        if novel_bias is not None:
            reg += lmbd * torch.norm(self._fc.bias[rng1:rng2, :] - novel_bias)**2
        return reg

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
