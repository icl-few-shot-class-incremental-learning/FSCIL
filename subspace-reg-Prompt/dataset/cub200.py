# This script is largely based on https://github.com/WangYueFt/rfs

import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import re

# torch.multiprocessing.set_sharing_strategy('file_system')
class cub200(Dataset):
    #index_path -> txt_path, 
    #index -> base size
    def __init__(self, args,root='./cub', train=True,
                 index_path=None, index=None, base_sess=None,transform=None,):
        super(Dataset, self).__init__()
        self.root = root
        self.base_sess = base_sess
        self.transform = transform
        self.index_path = index_path
        self.index = index

        self.train = train  # training set or test set
        self._pre_operate()
        
        #! HSJ self.mean, self.std original
        #self.mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        #self.std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        self.mean = [0.485,0.456,0.406]
        self.std = [0.229,0.224,0.225]
        
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.unnormalize = transforms.Normalize(mean=-np.array(self.mean)/self.std, std=1/np.array(self.std))

        if transform is None:
            if self.base_sess == True:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomResizedCrop(224),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                self.normalize
                ])
        else:
            self.transform = transform

        if self.train:
            # self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
            #base는 100까지 따라서 index = 100
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            #novel session에 대한 세션 정보 줘야함
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            if base_sess:
                print(index)
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                #modifying
                #novel, test
                self.data, self.targets = self.SelectfromNovelClasses(self.data, self.targets, index_path)
                       
            #HSJ self.labels
        self.labels = self.targets
        self.imgs = self._getImg(self.data)
            #HSJ self.imgs
        #HSJ LABELTOHUMAN

        # Labels are available by codes by default. Converting them into human readable labels.
        self.label2human =[""] *200
        with open('./cub/CUB_200_2011/' +'classes.txt', 'r') as f:
            for line in f.readlines():
                catname, humanname = line.strip().lower().split(' ')
                num,humanname = humanname.strip().lower().split('.')
                humanname = " ".join(humanname.split('_'))
                if int(catname) in range(1,201):
                    self.label2human[int(catname)-1]= humanname
        #HSJ LABELTOHUMAN

        #HSJ basec_map
        basec = np.sort(np.arange(100))
                
        # Create mapping for base classes as they are not consecutive anymore.
        self.basec_map = dict(zip(basec, np.arange(len(basec))))
        #HSJ basec_map

    def _getImg(self,d_list):
        img_list = []
        for d_path in d_list:
            c_img = Image.open(d_path).convert('RGB')
            c_img = np.array(c_img)
            c_img_transformed = self.transform(c_img)
            img_list.append(c_img_transformed.numpy())
        img_list_np = np.array(img_list)
        return img_list_np

        
    def SelectfromTxt(self, data2label, index_path):
        index = open('./cub/CUB_200_2011/index_list/session_'+ str(index_path) + '.txt').read().splitlines()
        data_tmp = []
        targets_tmp = []
        for i in index:
            img_path = self.root + '/'+ i
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])

        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        #index or 1, index+1
        for i in range(index):
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp
    
    def SelectfromNovelClasses(self, data, targets,index_path):
        data_tmp = []
        targets_tmp = []
        #100 or 101
        for i in range(100+((index_path-2)*10),100+((index_path-1)*10)):
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])

        return data_tmp, targets_tmp
    def list2dict(self, list):
        dict = {}
        for l in list:
            s = l.split(' ')
            id = int(s[0])
            cls = s[1]
            if id not in dict.keys():
                dict[id] = cls
            else:
                raise EOFError('The same ID can only appear once')
        return dict
    
    def text_read(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                lines[i] = line.strip('\n')
        return lines
        
    def _pre_operate(self):
            image_file = './cub/'+ 'CUB_200_2011/images.txt'
            split_file = './cub/'+ 'CUB_200_2011/train_test_split.txt'
            class_file = './cub/'+ 'CUB_200_2011/image_class_labels.txt'
            id2image = self.list2dict(self.text_read(image_file))
            id2train = self.list2dict(self.text_read(split_file))  # 1: train images; 0: test iamges
            id2class = self.list2dict(self.text_read(class_file))
            train_idx = []
            test_idx = []
            for k in sorted(id2train.keys()):
                if id2train[k] == '1':
                    train_idx.append(k)
                else:
                    test_idx.append(k)

            self.data = []
            self.targets = []
            self.data2label = {}
            if self.train:
                for k in train_idx:
                    image_path = './cub/'+ 'CUB_200_2011/images/'+ str(id2image[k])
                    self.data.append(image_path)
                    self.targets.append(int(id2class[k]) - 1)
                    self.data2label[image_path] = (int(id2class[k]) - 1)

            else:
                for k in test_idx:
                    image_path = './cub/'+ 'CUB_200_2011/images/'+ str(id2image[k])
                    self.data.append(image_path)
                    self.targets.append(int(id2class[k]) - 1)
                    self.data2label[image_path] = (int(id2class[k]) - 1)
            self.targets = np.array(self.targets)
                    
    def __getitem__(self, item):
        if self.base_sess:
            img = self.imgs[item]
            target = self.targets[item] - min(self.labels)
            
            return img, target,item
        else:
            if self.train == True and self.base_sess and self.n_base_support_samples > 0:
                    assert self.n_base_support_samples > 0
                    # These samples will be stored in memory for every episode.
                    support_xs = []
                    support_ys = []
                    if self.fix_seed:
                        np.random.seed(item)
                    cls_sampled = np.random.choice(self.classes, len(self.classes), False)
                    
                    for idx, cls in enumerate(np.sort(cls_sampled)):
                        imgs = np.asarray(self.data[cls]).astype('uint8')
                        support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]),
                                                                  self.n_base_support_samples,
                                                                  False)
                        support_xs.append(imgs[support_xs_ids_sampled])
                        support_ys.append([cls] * self.n_base_support_samples)    
                    support_xs, support_ys = np.array(support_xs), np.array(support_ys)
                    num_ways, n_queries_per_way, height, width, channel = support_xs.shape
                    support_xs = support_xs.reshape((-1, height, width, channel))
                    if self.n_base_aug_support_samples > 1:
                        support_xs = np.tile(support_xs, (self.n_base_aug_support_samples, 1, 1, 1))
                        support_ys = np.tile(support_ys.reshape((-1, )), (self.n_base_aug_support_samples))
                    support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                    support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))

                    # Dummy query.
                    query_xs = support_xs
                    query_ys = support_ys
            else:
            
                if self.fix_seed:
                    np.random.seed(item)

                #몇개로 나눌지(cub는 의미 없음)
                """BytesWarning
                if self.disjoint_classes:
                    cls_sampled = self.classes[:self.n_ways] # 
                    self.classes = self.classes[self.n_ways:]
                else:
                    cls_sampled = np.random.choice(self.classes, self.n_ways, False)
                """
                cls_sampled = self.targets

                support_xs = []
                support_ys = []
                query_xs = []
                query_ys = []
                for idx, cls in enumerate(np.sort(cls_sampled)):
                    #support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
                    support_xs.append(self.imgs)
                    #support_xs.append(imgs[support_xs_ids_sampled])
                    lbl = idx
                    if self.eval_mode in ["few-shot-incremental-fine-tune"]:
                        lbl = cls
                    support_ys.append([lbl] * self.n_shots) #

                    #query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
                    #query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
                    query_xs.append(self.imgs)
                    #query_xs.append(imgs[query_xs_ids])
                    query_ys.append([lbl] * 30) #

                support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(query_xs), np.array(query_ys)
                num_ways, n_queries_per_way, height, width, channel = query_xs.shape

                query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
                query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))

                support_xs = support_xs.reshape((-1, height, width, channel))
                """
                if self.n_aug_support_samples > 1:
                    support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
                    support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
                """
                support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
                query_xs = query_xs.reshape((-1, height, width, channel))
                query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

                support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
                query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs.float(), support_ys, query_xs.float(), query_ys
            

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = 'data'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    """
    print(len(imagenet))
    print(imagenet.__getitem__(500)[0].shape)

    metaimagenet = MetaImageNet(args)
    print(len(metaimagenet))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)
    """