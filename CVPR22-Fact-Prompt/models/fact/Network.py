import argparse
 
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
#from models.official_efficientNet import *
from models.efficientNet import *
#from torchvision.models.efficientnet import *
from typing import Any, Callable, Optional, List, Sequence
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation

class Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.parameter.Parameter(nn.init.uniform_(torch.FloatTensor(10,5,1000),0,0.01))
        self.prompt_key = nn.parameter.Parameter(nn.init.uniform_(torch.FloatTensor(10,1000),0,0.01))

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

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        # prompt
        self.prompt_module = Prompt()
        
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            #self.encoder = resnet18(True, args)
            self.encoder = efficientnet_b0(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            #self.encoder = efficientnet_b0(False, args)
            #self.num_features = 512
            self.num_features = 1000
            
            # 변경
            self.num_features = 26000
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #print('### model.summary')
        #print(self.encoder)
        
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)

        self.dummy_orthogonal_classifier=nn.Linear(self.num_features, self.pre_allocate-self.args.base_class, bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        
        self.dummy_orthogonal_classifier.weight.data=self.fc.weight.data[self.args.base_class:,:]
        print(self.dummy_orthogonal_classifier.weight.data.size())
        
        print('self.dummy_orthogonal_classifier.weight initialized over.')

    def forward_metric(self, x):
        #print('### original x: ')
        #print(x)
        x = self.encode(x)
        res_prompt = self.prompt_module(x)
        prompted_x = res_prompt['prompted_embedding']

        # print('### after encode x: ')
        #print(x)
        if 'cos' in self.mode:
            #print('### after normalize x: ')
            #print(F.normalize(x, p=2, dim=-1).shape)
            x1 = F.linear(F.normalize(prompted_x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            #x1 = F.linear(F.normalize(x, p=2, dim=-1).unsqueeze(dim=0), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(prompted_x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            x = torch.cat([x1[:,:self.args.base_class],x2],dim=1)
            #print(x.shape)
            
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x, res_prompt

    def forpass_fc(self,x):
        x = self.encode2(x)
        print('### after encoder: ', x.shape)
        if 'cos' in self.mode:
            
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        #print('*** start to sun self.encoder() with x: '+str(x.shape))
        #print(x)
        x = self.encoder(x)
        #print('*** after run self.encoder(): '+str(x.shape))
        #print(x)
        #x = F.adaptive_avg_pool2d(x, 1)
        #print('*** after adaptive_agv_pooling: '+str(x.shape))
        #print(x)
        #x = x.squeeze(-1).squeeze(-1)
        #print('*** after squeezing: '+str(x.shape))
        #print(x)
        return x
    
    def encode2(self, x):
        #print('*** start to sun self.encoder() with x: '+str(x.shape))
        #print(x)
        x = self.encoder(x)
        res_prompt = self.prompt_module(x)
        prompted_x = res_prompt['prompted_embedding']
        #print('*** after run self.encoder(): '+str(x.shape))
        #print(x)
        #x = F.adaptive_avg_pool2d(x, 1)
        #print('*** after adaptive_agv_pooling: '+str(x.shape))
        #print(x)
        #x = x.squeeze(-1).squeeze(-1)
        #print('*** after squeezing: '+str(x.shape))
        #print(x)
        return prompted_x

    
    def pre_encode(self,x):
        
        if self.args.dataset in ['cifar100','manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200']:
            x = self.encoder.block0(x) #
            x = self.encoder.block1(x) #
            x = self.encoder.block2(x) #
            x = self.encoder.block3(x) #
            x = self.encoder.block4(x) #

            
        
        return x
        
    
    def post_encode(self,x):
        if self.args.dataset in ['cifar100','manyshotcifar']:
            
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200']:
            ## added
            
            
            x = self.encoder.block5(x) #
            x = self.encoder.block6(x) #
            x = self.encoder.block7(x) #
            x = self.encoder.last_block(x)

            ### add 

            #x = self.encoder.features(x)
            #print('after features: ')
           # print(x.shape)
            x = self.encoder.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.encoder.classifier(x)
            res_prompt = self.prompt_module(x)
            prompted_x = res_prompt['prompted_embedding']

        if 'cos' in self.mode:
            x = F.linear(F.normalize(prompted_x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input, res_prompt = self.forward_metric(input)
            # print(input)
            # print(res_prompt)
            return input, res_prompt
        elif self.mode == 'encoder':
            input = self.encode2(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode2(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

