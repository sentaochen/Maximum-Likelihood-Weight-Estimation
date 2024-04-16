# -*- coding: utf-8 -*-
import torch.nn as nn

from model.basenet import network_dict
from utils import globalvar as gl
import torch.nn.utils.spectral_norm as sn


class MODEL(nn.Module):

    def __init__(self, basenet, n_class, bottleneck_dim):
        super(MODEL, self).__init__()
        self.basenet = network_dict[basenet]()
        self.basenet_type = basenet
        self._in_features = self.basenet.len_feature()
        self.bottleneck = nn.Sequential(
            nn.Linear(self._in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.fc = nn.Sequential(
            sn(nn.Linear(bottleneck_dim, bottleneck_dim)),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            sn(nn.Linear(bottleneck_dim, n_class))
        )
        self.fc[0].weight.data.normal_(0, 0.01)
        self.fc[0].bias.data.fill_(0.0)

        self.fc[-1].weight.data.normal_(0, 0.01)
        self.fc[-1].bias.data.fill_(0.0)
    def forward(self, inputs):
        DEVICE = gl.get_value('DEVICE')
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        outputs = self.fc(features)
        return features, outputs
    
    def get_bottleneck_features(self, inputs):
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        return features

    def get_fc_features(self, inputs):
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        return self.fc(features)

