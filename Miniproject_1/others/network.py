from others.layers import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

# The deep network we use is based on UNet

class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=4, norm='bnorm'):
        super(UNet, self).__init__()

        # number of input, output and intermediate channels
        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        """
        Encoder 
        """

        self.enc1_1 = CNR(1 * self.nch_in,  1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc1_2 = CNR(1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool1 = Pooling(pool=2, type='avg')

        self.enc2_1 = CNR(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc2_2 = CNR(2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool2 = Pooling(pool=2, type='avg')

        self.enc3_1 = CNR(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc3_2 = CNR(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool3 = Pooling(pool=2, type='avg')

        self.enc4_1 = CNR(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.enc4_2 = CNR(8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.pool4 = Pooling(pool=2, type='avg')

        self.enc5_1 = CNR(8 * self.nch_ker, 2 * 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        """
        Decoder 
        """

        self.dec5_1 = DNR(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.upsample4 = UpSampling(pool=2, type='nearest')

        self.dec4_2 = DNR(2 * 8 * self.nch_ker, 8 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec4_1 = DNR(8 * self.nch_ker,     4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.upsample3 = UpSampling(pool=2, type='nearest')

        self.dec3_2 = DNR(2 * 4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec3_1 = DNR(4 * self.nch_ker,     2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.upsample2 = UpSampling(pool=2, type='nearest')

        self.dec2_2 = DNR(2 * 2 * self.nch_ker, 2 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec2_1 = DNR(2 * self.nch_ker,     1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])

        self.upsample1 = UpSampling(pool=2, type='nearest')

        self.dec1_2 = DNR(2 * 1 * self.nch_ker, 1 * self.nch_ker, kernel_size=3, stride=1, norm=self.norm, relu=0.0, drop=[])
        self.dec1_1 = DNR(1 * self.nch_ker,     1 * self.nch_out, kernel_size=3, stride=1, norm=[],        relu=0.1,  drop=[], bias=False)

    def forward(self, x):

        """
        Encoder 
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)

        """
        Decoder 
        """
        dec5 = self.dec5_1(enc5)

        upsample4 = self.upsample4(dec5)
        cat4 = torch.cat([enc4, upsample4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        upsample3 = self.upsample3(dec4)
        cat3 = torch.cat([enc3, upsample3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        upsample2 = self.upsample2(dec3)
        cat2 = torch.cat([enc2, upsample2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        upsample1 = self.upsample1(dec2)
        cat1 = torch.cat([enc1, upsample1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = dec1
        x = nn.Sigmoid()(x)
        # add sigmoid to make sure the output range satisfies the constraints
        return x


# Initialize the weights for the network
# Default set to 'normal', i.e. fill in the values with normal distribution
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal': 
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  


# call the initialize weights function to initialize the network
def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, init_gain=init_gain)
    return net

# Code adapted from Hanyoseob/Pytorch-Noise2Noise: Learning Image Restoration without Clean Data
