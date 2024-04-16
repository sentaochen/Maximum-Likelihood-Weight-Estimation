# -*- coding: utf-8 -*-
import os
import torch
from torchvision import datasets, transforms

from loader.data_list import ImageList

def load_data_for_PDA(root_dir, dataset, src_domain, tar_domain, phase, batch_size, net, label_sampler=None, args=None):
    crop_size = 224 if net != 'alexnet' else 227
    resize_size = 256
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])}


    list_root_dir = os.path.join(root_dir, 'list')
    data_root_dir = os.path.join(root_dir)
    # data_list file name
    labeled_source_list = os.path.join(list_root_dir, '{}.txt'.format(src_domain))
    unlabeled_target_list = os.path.join(list_root_dir, '{}.txt'.format(tar_domain + '_' + str(args.partial_classes_num)))
    
    if phase == 'train':
        if label_sampler == None:
            src_data = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'],args=args)
                
            src_loader = torch.utils.data.DataLoader(src_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                    num_workers=4)
            tar_data = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'],args=args)
            
            tar_loader = torch.utils.data.DataLoader(tar_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                    num_workers=4)
        else:
            src_data = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['train'],args=args)
                
            src_loader = torch.utils.data.DataLoader(src_data, batch_size=batch_size, sampler = label_sampler, drop_last=True,
                                                    num_workers=4)
            tar_data = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['train'],args=args)
            
            tar_loader = torch.utils.data.DataLoader(tar_data, batch_size=batch_size, shuffle=True, drop_last=True,
                                                    num_workers=4)
    else:
        src_data = ImageList(labeled_source_list, data_root_dir, transform=transform_dict['test'],args=args)
            
        src_loader = torch.utils.data.DataLoader(src_data, batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=4)
        tar_data = ImageList(unlabeled_target_list, data_root_dir, transform=transform_dict['test'],args=args)
        
        tar_loader = torch.utils.data.DataLoader(tar_data, batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=4)
    return src_loader, tar_loader

