# This script is largely based on https://github.com/WangYueFt/rfs

from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import time
import subprocess
import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.cub200 import cub200
from models.util import create_model
from dataset.mini_imagenet import MetaImageNet, ImageNet
from dataset.transform_cfg import transforms_test_options,transforms_options

from util import create_and_save_embeds
from eval.language_eval_copy_pretrained import few_shot_finetune_incremental_test
from configs import parse_option_eval

from efficientnet_pytorch import EfficientNet

from setproctitle import setproctitle
setproctitle('WG_SUBREG')

def main():

    opt = parse_option_eval()

    # Add git commit hash
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                               shell=False,
                               stdout=subprocess.PIPE)
    git_head_hash = process.communicate()[0].strip()
    opt.git_head_hash = git_head_hash.decode()

    # Set seeds
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)

    print("************* Training arguments *************")
    args = opt
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("End of arguments.\n")


    base_support_loader = None

    if opt.dataset == 'cub200':
        train_trans, test_trans = transforms_test_options[opt.transform]
        train_trans_train, test_trans_train = transforms_options[opt.transform]

        # Base test samples loader. "split=train" refers to the base set of classes
        # "phase=test" means we are interested in those samples that were not used in
        # training.
        base_test_loader = DataLoader(cub200(args=opt, base_sess = True, train = False, transform=test_trans,index = 100, index_path = 1),
                                  #test_base_batch_size(to next)
                                  batch_size=256, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)

        meta_valloader = DataLoader(cub200(args=opt, base_sess = False, train=False, transform=train_trans,index = 100, index_path = 1),
                                batch_size=64 , shuffle=False, drop_last=False,
                                num_workers=opt.num_workers)
        
        if opt.n_base_support_samples > 0:
            ''' We'll use support samples from base classes. '''
            base_support_loader = DataLoader(cub200(args=opt, base_sess = True, train = True, transform=train_trans_train,index = 100, index_path = 1),
                                  batch_size=100,shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)


        

        # Test samples from novel classes as they are introduced.
        # split=val means we are interested in novel classes.
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)


    # Load model if available, check bias.
    ckpt = torch.load(opt.model_path)

    vocab = None
    # In this scenario we'll need the label embeds saved.
    # Label pull is used interchangeably with semantic subspace reg.
    if opt.label_pull is not None: # label_pull refers to gamma in the paper.
        vocab_train = [name for name in base_test_loader.dataset.label2human[0:100]]
        vocab_val = [name for name in meta_valloader.dataset.label2human[100:]]
        vocab_all = vocab_train + vocab_val # + vocab_test
        create_and_save_embeds(opt, vocab_all)

    # Linear layer bias is determined based on backbone.
    # Warning: we often assumed no linear bias.
    
    if opt.classifier =="linear":
        if 'classifier.bias' in ckpt['model'].keys():
            if ckpt['model']['classifier.bias'] is None:
                raise ValueError()
            opt.linear_bias = True
        else:
            opt.linear_bias = False

    # Load model.
    #? HSJ 
    #model = create_model(opt.model, n_cls, opt, vocab=vocab, dataset=opt.dataset)

    #pre-trained model load(ckpt가 학습된 모델이므로 해당 정보를 불러온 pretrained model에 옮겨 저장)
    model = EfficientNet.from_pretrained('efficientnet-b0',num_classes = 100)
    # print(model)
    model._fc.bias = None
    model.add_prompt()
    # print(model)
    # print(ckpt['model'])
    # print(ckpt['model']['_fc.weight'])
    # print(ckpt['model']['_fc.weight'].shape)
    # ss
    print("Loading model...")
    model.load_state_dict(ckpt['model'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True
    # print(model)
    # ss
    # Evaluation
    assert opt.classifier == "linear"
    criterion = nn.CrossEntropyLoss()

    
    start = time.time()
    opt.split = "val"
    opt.neval_episodes = 10 # If multi-session, this is overridden later.
    novel, base = few_shot_finetune_incremental_test(model,
                                                        ckpt,
                                                        criterion,
                                                        meta_valloader,
                                                        base_test_loader,
                                                        opt,
                                                        base_support_loader=base_support_loader)
    val_time = time.time() - start
    avg_score = (base+novel)/2
    print('val_acc_novel: {:.4f}, std: {:.4f}, time: {:.1f}'.format(novel, 0, val_time))
    print('val_acc_base: {:.4f}, std: {:.4f}, time: {:.1f}'.format(base, 0, val_time))
    print('val_acc_average: {:.4f}'.format(avg_score))



if __name__ == '__main__':
    main()