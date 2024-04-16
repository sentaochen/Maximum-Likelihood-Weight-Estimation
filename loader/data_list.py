# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image

class ImageList(torch.utils.data.Dataset):
    def __init__(self, file_name, root_dir, transform=None, type=None,args=None):
        super(ImageList, self).__init__()
        self.transform = transform
        self.type = type
        self.images, self.labels = [], []
        self.class_labels = []
        self.pseudo_labels = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for item in lines:
                line = item.strip().split(' ')
                self.images.append(os.path.join(root_dir, line[0]))
                self.labels.append(int(line[1].strip()))
        self.class_num = args.class_num
        self.class_labels = [ [] for i in range(self.class_num)]
        for idx, label in enumerate(self.labels):
            self.class_labels[label].append(idx)

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.images)
    