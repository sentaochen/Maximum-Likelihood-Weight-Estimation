# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import time
import math
from utils.optimizer import get_optimizer
from utils import globalvar as gl
from utils.get_weight import get_weight

from loss.loss import weighted_cross_entropy, weighted_smooth_cross_entropy

def entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def test(loader, model):
    DEVICE = gl.get_value('DEVICE')
    model.eval()
    size = len(loader.dataset)
    start_test = True
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            features, outputs = model(inputs)
            if start_test:
                all_features = features.float()
                all_outputs = outputs.float()
                all_labels = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features,features.float()),0)
                all_outputs = torch.cat((all_outputs, outputs.float()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)
        all_preds = torch.max(all_outputs, 1)[1]
        correct = torch.sum(all_preds == all_labels.data).item()
        acc = 100.0 * (float)(correct) / size
    return acc, all_features, all_preds, all_labels



def train_for_PDA(args, model, dataloaders):
    DEVICE = gl.get_value('DEVICE')
    record_file = gl.get_value('record_file')
    if args.early:
        best_acc = 0
        counter = 0
    src_data_l = iter(dataloaders['src_train_l'])
    tar_data_ul = iter(dataloaders['tar_train_ul'])
    
    start_time = time.time()
    weights = torch.ones(len(dataloaders["src_train_l"].dataset)).to(DEVICE)
    for step in range(1, args.steps + 1):
        optimizer = get_optimizer(model, args.lr, args.lr_mult, args.momentum, args.decay, step = step)
        model.train(True)

        if step > 0 and step % len(dataloaders['src_train_l']) == 0:
            src_data_l = iter(dataloaders['src_train_l'])
        if step > 0 and step % len(dataloaders['tar_train_ul']) == 0:
            tar_data_ul = iter(dataloaders['tar_train_ul'])


        inputs_l, labels_l, indexs = next(src_data_l)
        inputs_ul, _, _ = next(tar_data_ul)
        s_img, s_label = inputs_l.to(DEVICE), labels_l.to(DEVICE)
        t_img = inputs_ul.to(DEVICE)
        s_features, s_output = model(s_img)
        t_features, t_output = model(t_img)
        softmax_layer = nn.Softmax(dim=1).to(DEVICE)
        t_softmax = softmax_layer(t_output)
        tar_loss = torch.mean(entropy(t_softmax))
        weight = weights[indexs].to(DEVICE)
        cls_loss = weighted_cross_entropy(s_output, s_label, weight)
        # cls_loss = weighted_smooth_cross_entropy(s_output, s_label, weight) # label smoothing
        if step < args.start_update:
            loss = cls_loss
        else:
            lambd = 2 / (1 + math.exp(-10 * (step - args.start_update) / args.steps)) - 1
            loss = cls_loss + lambd * tar_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        print('Step: [{}/{}]:  cls_loss:{:.4f}   tar_loss:{:.4f}'.format(step, args.steps, loss.item(), tar_loss.item()))
        if step % 1000 == 0:
            acc_src, all_source_features, _, all_source_labels = test(dataloaders['src_test'], model)
            acc_tar, all_target_features, all_target_pseudo_labels, _ = test(dataloaders['tar_test'], model)
            print('acc on source: {}, acc on target_test: {} '.format(acc_src, acc_tar))
        if step > 0 and step % args.save_interval == 0:
            print('{} step train time: {:.1f}s'.format(args.save_interval, time.time()-start_time))
            test_time = time.time()

            acc_src, all_source_features, _, all_source_labels = test(dataloaders['src_test'], model)
            acc_tar, all_target_features, all_target_pseudo_labels, _ = test(dataloaders['tar_test'], model)
            print('acc on source: {}, acc on target: {} '.format(acc_src, acc_tar))
            print('one test time: {:.1f}s'.format(time.time()-test_time))
            print('record {}'.format(record_file))
            if args.early:
                if acc_tar > best_acc:
                    best_acc = acc_tar
                    counter = 0
                else:
                    counter += 1
                    if counter > args.patience:
                        print('early stop! training_step:{}'.format(step))
                        break
            seconds = time.time() - start_time
            print('{} step cost time: {}h {}m {:.0f}s\n'.format(step, seconds//3600, seconds%3600//60, seconds%60))
            
            if step >= args.start_update and step % args.update_weight == 0:
                weights = get_weight(all_source_features.cpu(), all_source_labels.cpu(), all_target_features.cpu(), all_target_pseudo_labels.cpu())
                weights = torch.Tensor(weights[:])
                print(torch.sum(weights[torch.nonzero(all_source_labels<args.partial_classes_num)]))
                print(torch.sum(weights[torch.nonzero(all_source_labels>args.partial_classes_num)]))

    time_pass = time.time() - start_time
    with open(record_file, 'a') as f:
        f.write('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training {} step complete in {}h {}m {:.0f}s\n'.format(step, time_pass//3600, time_pass%3600//60, time_pass%60))
    print('Training_step:{}, best_acc:{}'.format(step, best_acc))

