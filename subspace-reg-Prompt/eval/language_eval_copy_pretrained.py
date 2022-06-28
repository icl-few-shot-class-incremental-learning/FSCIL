# This script is partially based on https://github.com/WangYueFt/rfs

from __future__ import print_function
import numpy as np
import copy
import itertools

from dataset.cub200 import cub200
from torch.utils.data import DataLoader
import torch
from .util import accuracy, image_formatter,\
get_vocabs, drop_a_dim,drop_a_dim_t,\
get_optim, freeze_backbone_weights,\
AverageMeter, log_episode
from dataset.memory import Memory
from models.resnet_language import LangPuller
import pandas as pd

from dataset.transform_cfg import transforms_options, transforms_list,transforms_test_options

#get_vocabs, drop_a_dim,drop_a_dim_t,\
def validate(query_xs, query_ys_id, net, criterion, opt, epoch):
    net.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            if isinstance(query_xs, list):
                acc1, acc5, losses, preds = [],[],[],[]
                for i,item in enumerate(query_xs):
                    r = validate(item, query_ys_id[i], net, criterion, opt, epoch)
                    acc1.append(r[0])
                    acc5.append(r[1])
                    losses.append(r[2])
                    preds.append(r[3])
                return acc1, acc5, losses, preds
            else:
                query_xs = query_xs.cuda()
                query_ys_id = query_ys_id.cuda()

                # compute output
                output, res_prompt = net(query_xs)
                loss = criterion(output, query_ys_id)
                
                sur_loss = res_prompt["reduce_sim"]
                # loss = loss+0.5*sur_loss
                # loss = loss+0.1*sur_loss
                # loss = loss-0.5*sur_loss
                loss = loss-0.1*sur_loss

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, query_ys_id, topk=(1, 5))
                query_ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()

                return acc1[0], acc5[0], loss.item(), query_ys_pred


def eval_base(net, base_batch, criterion, vocab_all=None, df=None, return_preds=False):
    acc_base_ = []
    net.eval()
    with torch.no_grad():
        input, target, *_ = base_batch
        input = input.squeeze(0).cuda()
        target = target.squeeze(0).cuda()
        output, res_prompt = net(input)
        acc1, _ = accuracy(output, target, topk=(1, 5))
        acc_base_.append(acc1[0].item())
        if return_preds:
            ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()

        if df is not None:
            ys_pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            imgdata = input.detach().cpu().numpy()
            base_info = [(0, vocab_all[target[i]], True, vocab_all[ys_pred[i]],
                          image_formatter(imgdata[i,:,:,:])) for i in range(len(target))]
            df = df.append(pd.DataFrame(base_info, columns=df.columns), ignore_index=True)

        if return_preds:
            return np.mean(acc_base_), ys_pred

    return np.mean(acc_base_)

def few_shot_finetune_incremental_test(net,
                                       ckpt,
                                       criterion,
                                       meta_valloader,
                                       base_val_loader,
                                       opt,
                                       vis=False,
                                       base_support_loader=None):

    # Create dataframes to collect data for visualization.
    if vis:
        cols = ['idx', 'class', 'isbase', 'predicted', 'img']
        df = pd.DataFrame(columns=cols)
    if opt.track_weights:
        cols = ["episode", "type", "label", "class",
                "fine_tune_epoch", "classifier_weight"]
        track_weights = pd.DataFrame(columns=cols)
    if opt.track_label_inspired_weights:
        cols = ["episode", "label", "fine_tune_epoch", "inspired_weight"]
        track_inspired = pd.DataFrame(columns=cols)


    train_trans, test_trans = transforms_test_options[opt.transform]

    # Create meters.
    acc_novel, acc_base = [AverageMeter() for _ in range(2)]
    weighted_avg_l, acc_novel_list, acc_base_list = [[] for _ in range(3)]

    # Used for creation of confusion matrices.
    if opt.save_preds_0:
        preds_df = pd.DataFrame(columns=["Episode", "Gold", "Prediction"])

    # Reset seeds.
    torch.manual_seed(opt.set_seed)
    np.random.seed(opt.set_seed)

    # Pretrained backbone, if not continual,
    # net will be reset before every episode.
    basenet = copy.deepcopy(net).cuda()
    base_weight, base_bias = basenet._get_base_weights()

    # Loaders for fine tuning and testing.
    #base_valloader_it 은 base test 셋
    base_valloader_it = itertools.cycle(iter(base_val_loader))
    #meta는 사용하지 않음
    meta_valloader_it = itertools.cycle(iter(meta_valloader))
    if base_support_loader is not None:
        base_support_it = itertools.cycle(iter(base_support_loader))
        # Use the same set of examples for every episode.
        # i.e. keep a fixed set of base examplars in memory.
        base_support_xs, base_support_ys, *_ = drop_a_dim_t(next(base_support_it))

    # Collect a set of query samples.
    novel_query_collection = None
    novel_query_collection_id = None
    #base_test_valloader 값 [imgs,targets,cat2label순으로 넣기]
    base_batch = next(base_valloader_it)  # We use the same large base batch every time for eval.
    
    # Create memory
    if opt.memory_replay:
        memory = Memory()
        
    # Initial validation on base samples.
    acc_base_ = eval_base(net, base_batch, criterion)
    weighted_avg_l.append(acc_base_)

    # How many episodes/sessions?
    iter_num = opt.neval_episodes

    # If multi-session (continual), we need 8 sessions for miniImageNet
    if opt.continual:
        #! HSJ iter_num=8 to iter_num =10(session 10)
        iter_num = 10  # Assumes miniImageNet.

        # Retrieve the classes used for base training.
        basec_map = ckpt['training_classes']
        basec_map_rev = {}
        for k, v in basec_map.items():
            basec_map_rev[v] = k

    # Iterate over sessions.
    for idx in range(iter_num):
        print("\n**** Iteration {}/{} ****\n".format(idx+1, opt.neval_episodes))
        #d_idx = drop_a_dim(next(meta_valloader_it))
        val_train = DataLoader(cub200(args=opt,base_sess = False, train=True,transform = train_trans,index_path = (idx+2)),batch_size = 64,shuffle = False, drop_last=False, num_workers = opt.num_workers)

        val_test = DataLoader(cub200(args=opt,base_sess = False, train=False,transform = test_trans, index_path = (idx+2)),batch_size = 64,shuffle = False, drop_last=False, num_workers = opt.num_workers)

        #HSJ
        #after drop_a_dim
        #support_xs = train image(novel) (maybe) size (50,3,224,224) type tensor
        #support_ys = train label(novel) (maybe)size 50 type numpy_array
        #query_xs = test image(novel) (maybe)size (300,3,224,224) type tensor
        #query_ys = test label(novel) (maybe)size 300
        support_xs = val_train.dataset.imgs
        support_ys = val_train.dataset.targets
        query_xs = val_test.dataset.imgs
        query_ys = val_test.dataset.targets

        support_xs = torch.Tensor(support_xs)
        
        support_xs , support_ys , query_xs, query_ys = drop_a_dim(support_xs, support_ys, query_xs, query_ys)

        if base_support_loader is not None:
            
            support_xs = torch.cat([support_xs, base_support_xs], 0)
            
        if vis:
            novelimgs = query_xs.detach().numpy()
            
        
        # Get vocabs for the loaders.
        if idx > 0:
            prev_vocab_base = vocab_base
            prev_vocab_novel = vocab_novel
        #HSJ modify
        out_vocabs = get_vocabs(val_train, val_test, query_ys)
        vocab_base, vocab_all, vocab_novel, orig2id = out_vocabs
        print("Vocab base: ", vocab_base)
        print("Vocab novel: ", vocab_novel)

        if idx == 0:
            orig_base_num = len(vocab_base)
        if idx > 0:
            vocab_base = prev_vocab_base + prev_vocab_novel

        # We regularize the change in novel weights based on their value
        # at the end of the session in which they were first introduced.
        # Here we save those values.
        if idx == 1:
            #! HSJ modify classifier to _fc
            #! HSJ modify detach()[-opt.n_ways to 10]
            #이번에 학습한 novel weight를 저장하고 추후에 reg_loss_novel을 계산할 때 보존하도록 사용함 novel의 학습률(base 보존하듯이)
            novel_weight_to_reserve = net._fc.weight.clone().detach()[-opt.n_ways:,:].requires_grad_(False)
            novel_bias_to_reserve = None
            if base_bias is not None:
                #! HSJ modify classifier to _fc
                novel_bias_to_reserve = net._fc.bias.clone().detach()[-opt.n_ways:].requires_grad_(False)
            print(f"Novel weight to reserve is of shape {novel_weight_to_reserve.shape} at session {idx+1}.")
        if idx > 1:
            #! HSJ modify classifier to _fc
            new_novel_set = net._fc.weight.clone().detach()[-opt.n_ways:,:].requires_grad_(False)
            novel_weight_to_reserve = torch.cat((novel_weight_to_reserve, new_novel_set), 0)
            if base_bias is not None:
                #! HSJ modify classifier to _fc
                new_novel_set_bias = net._fc.bias.clone().detach()[-opt.n_ways:].requires_grad_(False)
                novel_bias_to_reserve = torch.cat((novel_bias_to_reserve, new_novel_set_bias), 0)

            print(f"Novel weight to reserve is of shape {novel_weight_to_reserve.shape} at session {idx+1}.")


        # Get sorted numeric labels
        novel_labels = np.sort(np.unique(query_ys))  # True labels.
        print("Novel labels: ", novel_labels)

        # Map the labels to their new form.
        for k, v in orig2id.items():
            orig2id[k] = v + idx*opt.n_ways
        query_ys_id = torch.LongTensor([orig2id[y] for y in query_ys])
        support_ys_id = torch.LongTensor([orig2id[y] for y in support_ys])

        # Add the new set of queries to the collection.
        if novel_query_collection_id is None:
            novel_query_collection = [query_xs]
            novel_query_collection_id = [query_ys_id]
        else:
            novel_query_collection.append(query_xs)
            novel_query_collection_id.append(query_ys_id)
            

        if base_support_loader is not None:
            
            support_ys_id = torch.cat([support_ys_id,
                                       torch.from_numpy(base_support_ys)])

        net.train()

        # Augment the net's classifier to accommodate new classes.
        net.augment_base_classifier_(len(novel_labels))

        # Label pulling is a regularization towards the label attractors.
        # In the paper, we call this operation semantic subspace reg.
        #HSJ test and False 추가
        if opt.label_pull is not None and opt.pulling == "regularize":
            if idx == 0:
                lang_puller = LangPuller(opt, vocab_base, vocab_novel)
            else:
                # Augment the last layer 
                lang_puller.update_novel_embeds(vocab_novel)

            if opt.attraction_override == "mapping_linear_label2image":
                lang_puller.create_pulling_mapping(ckpt[opt.attraction_override])

            pullers = lang_puller(base_weight[:orig_base_num, :])

        # Optimizer
        optimizer = get_optim(net, opt)  # TODO anything to load from ckpt?

        # Fine tuning epochs.
        train_loss = 15
        epoch = 1

        # Stable epochs
        opt.stable = True if opt.target_train_loss == 0 else False
        stable_epochs = 0

        stop_condition = True
        while stop_condition:
            #_fc -> model classifier , weight의 require_grad 설정\
            freeze_backbone_weights(net, opt, epoch, exclude=["_fc","fc","prompt"])
            support_xs = support_xs.cuda()
            support_ys_id = support_ys_id.cuda()

            # Compute output
            # !
            if opt.classifier in ["lang-linear", "description-linear"] and opt.attention is not None:
                output, alphas = net(support_xs, get_alphas=True)
                loss = criterion(output, support_ys_id) + opt.diag_reg * criterion(alphas, support_ys_id)
            else:
                output, res_prompt = net(support_xs)
                loss = criterion(output, support_ys_id)
                sur_loss = res_prompt["reduce_sim"]
                # loss = loss+0.5*sur_loss
                # loss = loss+0.1*sur_loss
                # loss = loss-0.5*sur_loss
                loss = loss-0.1*sur_loss

                
            # Memory replay
            if opt.memory_replay and len(memory) > 0:
                output_, _res_prompt = net(memory.data)
                loss += criterion(output_, memory.labels)
                sur_loss = _res_prompt["reduce_sim"]
                # loss = loss+0.5*sur_loss
                # loss = loss+0.1*sur_loss
                # loss = loss-0.5*sur_loss
                loss = loss-0.1*sur_loss

          
            # Penalize the change in base classifier weights.
            #reg를 loss에 더함으로써 모델 구조 만드는데 여기에 들어가는 파라미터들 조정이 필요함
            if opt.lmbd_reg_transform_w is not None:
                lmbd_reg = net.regloss(opt.lmbd_reg_transform_w, base_weight, base_bias)
                # if epoch % 10 == 0:
                #     print("LMBD: ", lmbd_reg.item())
                loss += lmbd_reg

            # Penalize the change in previous novel classifier weights.
            if opt.lmbd_reg_novel is not None and idx > 0:
                lmbd_reg2 = net.reglossnovel(opt.lmbd_reg_novel,
                                             novel_weight_to_reserve,
                                             novel_bias_to_reserve)
                # if epoch % 10 == 0:
                #     print("LMBDN: ", lmbd_reg.item())
                loss += lmbd_reg2

            # Subspace regularizer loss
            if opt.label_pull is not None and opt.pulling == "regularize":

                # Default is semantic subspace reg, here override language
                # pulling i.e. regularization with simple subspace regularizer
                #distance2subspace를 뒤에 t 추가해서 무시함
                #! HSJ modify classifier to _fc
                if opt.attraction_override == "distance2subspace":
                    pullers = lang_puller.get_projected_weight(base_weight,
                                                               net._fc.weight[len(vocab_base):,:])
                #test HsJ 일단 주석처리 to line 318
                reg = lang_puller.loss1(opt.label_pull,
                                        pullers,
                                        net._fc.weight[len(vocab_base):,:])
                if epoch % 10 == 0:
                    print("PULL: ", reg.item())
                loss += reg

            # Train
            #원규
            optimizer.zero_grad()
            #! HSJ requires_gra
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                # Check if training converges
                if opt.stable:
                    if abs(loss.item() - train_loss) < opt.convergence_epsilon:
                        stable_epochs += 1
                    else:
                        stable_epochs = 0
                    if stable_epochs == opt.stable_epochs: stop_condition = False


                acc1, acc5 = accuracy(output, support_ys_id, topk=(1,5))
                train_loss = loss.item()
                if epoch % 10 == 0:
                    print('Novel Epoch {:4d}\t'
                          'Train Loss {:10.4f}\t'
                          'Acc@1 {:10.3f}\t'
                          'Acc@5 {:10.3f}'.format(
                           epoch, train_loss, acc1[0], acc5[0]))

                if (epoch >= opt.max_novel_epochs) or (train_loss <= opt.target_train_loss and epoch >= opt.min_novel_epochs + 1):
                    stop_condition = False

            test_acc, test_acc_top5, test_loss, query_ys_pred = validate(novel_query_collection,
                                                                                  novel_query_collection_id,
                                                                                  net,
                                                                                  criterion,
                                                                                  opt,
                                                                                  epoch)
            #print('query_ys_pred', query_ys_pred)
            #print('novel_query_collection_id',novel_query_collection_id)

           # class_acc=0
           # for i in range(len(query_ys_pred)):
           #     if query_ys_pred[i]==novel_query_collection_id[i]:
           #         class_acc+=1


            if opt.track_label_inspired_weights:
                inspired_weights = label_inspired_weights.clone().cpu().numpy() #fixme
                for k,lbl in enumerate(vocab_novel):
                    track_inspired.loc[len(track_inspired)] = [idx, lbl, epoch, inspired_weights[k]]


            if opt.track_weights:
                #! HSJ modify classifier to _fc
                classifier_weights = net._fc.weight.clone().detach().cpu().numpy()
                for k,lbl in enumerate(vocab_base):
                    track_weights.loc[len(track_weights)] = [idx, "base", lbl, vocab_base[k],
                                                             epoch, classifier_weights[k]]
                len_base = len(vocab_base)
                for k,lbl in enumerate(vocab_novel):
                    track_weights.loc[len(track_weights)] = [idx, "novel", lbl, vocab_novel[k],
                                                             epoch, classifier_weights[len_base+k]]

            

            if vis and idx == 0:
                novel_info = [(idx, vocab_all[query_ys_id[i]], False, vocab_all[np.array(query_ys_pred[0])[i]],
                               image_formatter(novelimgs[i,:,:,:]))  for i in range(len(query_ys_id))]
                               
                df = df.append(pd.DataFrame(novel_info, columns=df.columns), ignore_index=True)

            epoch += 1

        # Sample from this session's support and augment memory for replay.
        if opt.memory_replay:
            inds = np.random.choice(opt.n_shots, opt.memory_replay)
            margin = 5 * np.arange(5)
            offset = np.arange(0,125,25)
            inds = np.tile(margin + inds, (5,1)) + (np.tile(offset,(5,1))).T
            inds = inds.flatten()
            memory.additems(support_xs[inds[0:10], :], support_ys_id[inds[0:10]])
            
        # Evaluate base samples with the updated network
        vis_condition = (vis and idx == 0)
        acc_base_ = eval_base(net,
                              base_batch,
                              criterion,
                              vocab_all = vocab_all if vis_condition else None,
                              df= df if vis_condition else None)

        
        if isinstance(test_acc, list):
            # Report accuracies from first session towards last
            test_acc = [round(i.item(),2) for i in test_acc]#############클래스별테스트acc 출력
            print("Novel session accuracies: ", test_acc)
            test_acc = np.array(test_acc).mean()
        else:
            test_acc = test_acc.item()
            
        # Update meters.
        acc_base.update(acc_base_)
        acc_novel.update(test_acc)
        
        # Number of base classes
        #w1 = 60 if opt.dataset == "miniImageNet" else 200 # tiered

        w1 = 100 if opt.dataset =='cub200' else 200
        # Number of novel classes
        w2 = len(vocab_base) + len(vocab_novel) - 100

        # Accuracy is a weighted average according to the total
        # number of classes in each category (base and novel).
        weighted_avg = (w1*acc_base_ + w2*test_acc)/(w1+w2)
        weighted_avg_l.append(round(weighted_avg,2))
        acc_novel_list.append(round(test_acc,2))
        acc_base_list.append(round(acc_base_,2))

        print(f"***Running weighted avg: {weighted_avg}")

        # Log episode results.
        log_episode(novel_labels,
                    vocab_novel,
                    epoch,
                    test_acc,
                    acc_base_,
                    acc_base.avg,
                    acc_novel.avg)

        
        if opt.save_preds_0:
        # Saving predictions for visualization/error analysis, if specified.
            _, base_query_ys, *_ = base_batch
            base_query_ys = base_query_ys.squeeze(0)
            acc_base_, base_preds = eval_base(net, base_batch, criterion,
                                              vocab_all, return_preds=True)
            if idx == 0:
                id2orig = {}
            for k, v in orig2id.items():
                id2orig[v] = k

            # base_size = net.num_classes
            query_ys_pred_orig, novel_collective_ys_orig = map2original([query_ys_pred[0],
                                                               novel_query_collection_id[0]],
                                                              [id2orig,
                                                               basec_map_rev])
            base_preds_orig, base_query_ys_orig = map2original([base_preds,
                                                      base_query_ys],
                                                     [id2orig,
                                                      basec_map_rev])


            temp_df = pd.DataFrame({
                 "Episode": np.repeat(idx, len(novel_collective_ys_orig)+len(base_query_ys_orig)),
                 "Gold": np.concatenate((novel_query_collection_id[0], base_query_ys), 0),
                 "Prediction": np.concatenate((query_ys_pred[0], base_preds), 0).astype(int),
                 "Original_Gold": np.concatenate((novel_collective_ys_orig, base_query_ys_orig), 0),
                 "Original_Prediction": np.concatenate((query_ys_pred_orig, base_preds_orig), 0).astype(int)})
            preds_df = pd.concat([preds_df, temp_df], 0)
            if idx == iter_num-1:
                filename = f"csv_files_mem/seed_{opt.set_seed}_{opt.dataset}_{opt.n_shots}_{opt.label_pull}_{opt.attraction_override}_continual_{opt.continual}_mem_{opt.memory_replay}_predictions.csv"
                preds_df.to_csv(filename, index=False)


    # Tracking old and new parameters for PCA visualization.
    if opt.track_label_inspired_weights:
        track_inspired.to_csv(f"track_inspired_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if opt.track_weights:
        track_weights.to_csv(f"track_weights_{opt.eval_mode}_pulling_{opt.pulling}_{opt.label_pull}_target_loss_{opt.target_train_loss}_synonyms_{opt.use_synonyms}.csv", index=False)

    if vis:
        return df
    else:
        print("Overall continual accuracies: ", weighted_avg_l)
        print("Novel only incremental: ", acc_novel_list)
        print("Base only incremental: ", acc_base_list)
        return acc_novel.avg, acc_base.avg


def map2original(ls, dictlist):
    combined = {}
    for d in dictlist:
        for k, v in d.items():
            if k in combined:
                raise ValueError()
            else:
                combined[k] = v
    values = combined.values()
    assert len(np.unique(values)) != len(values)
    rlist = []*len(ls)
    for l0 in ls:
        if not isinstance(l0, list):
            l0 = l0.tolist()
        rlist.append([combined[el] for el in l0])
    return rlist
