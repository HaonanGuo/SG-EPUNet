import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
import sys
sys.path.append('/public/home/ghn/BuildingChange_Pytorch')
from core.encoder import Resnet

nonlinearity=nn.LeakyReLU
activaionF=nn.functional.leaky_relu
class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 n_filters=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        if kernel_size == 3:
            conv_padding = 1
        elif kernel_size == 1:
            conv_padding = 0

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels,
                               in_channels // 4,
                               kernel_size,
                               padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,
                                              in_channels // 4,
                                              3,
                                              stride=2,
                                              padding=1,
                                              output_padding=conv_padding, bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,
                               n_filters,
                               kernel_size,
                               padding=conv_padding, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class ResNet34Unet(nn.Module):
    def __init__(self,
                 num_classes,
                 criterion=None,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3
                 ):
        super().__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        # self.firstconv = resnet.conv1
        # assert num_channels == 3, "num channels not used now. to use changle first conv layer to support num channels other then 3"
        # try to use 8-channels as first input
        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)

        self.finalconv = nn.Sequential(nn.Conv2d(filters[0], 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))

        self.criterion=nn.BCEWithLogitsLoss()
    def cal_loss(self,seg,gts):
        return self.criterion(seg.squeeze(), gts.squeeze().float())
    def forward(self, x,gts):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))
        f = self.finalconv(d1)
        return self.criterion(f.squeeze(), gts[0].squeeze().float()),[f,f]
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = activaionF
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Attetion(nn.Module):
    def __init__(self, inchannels, kernel_size=7):
        super(Attetion, self).__init__()
        self.CA = ChannelAttention(in_planes=inchannels)
        self.SA = SpatialAttention(kernel_size=kernel_size)

    def forward(self, input):
        out = self.CA(input) * input
        out = self.SA(out)
        return out
class PSPModule(nn.Module):
    def __init__(self, features, out_features=64, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)
class EPUNet (nn.Module):
    def __init__(self,
                 num_classes,
                 criterion,
                 num_channels=3,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()
        self.criterion=criterion
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}
        self.psp=PSPModule(8+64)
        self.firstconv = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(inplace=True),
                                                        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1,1),
                                                                  padding=(1, 1), bias=False),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(inplace=True),

                                                        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1),
                                                                  padding=(1, 1), bias=False),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(inplace=True),
                                                        )


        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.center = DecoderBlock(in_channels=filters[3],
                                   n_filters=filters[3],
                                   kernel_size=decoder_kernel_size,
                                   is_deconv=is_deconv)
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[2],
                                     n_filters=filters[2],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention4=Attetion(filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[1],
                                     n_filters=filters[1],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention3 = Attetion(filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention2 = Attetion(filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],
                                     n_filters=filters[0],
                                     kernel_size=decoder_kernel_size,
                                     is_deconv=is_deconv)
        self.attention1 = Attetion(filters[0])
        self.finalconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))
        self.midconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, num_classes, 1))
        self.res4 = Resnet.BasicBlock(64, 64, stride=1, downsample=None)
        self.conv4=nn.Conv2d(64,32,1)
        self.res3 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.conv3 = nn.Conv2d(32, 16, 1)
        self.res2 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.conv2 = nn.Conv2d(16, 8, 1)
        self.res1 = Resnet.BasicBlock(8, 8, stride=1, downsample=None)
        self.conv1 = nn.Conv2d(8, 1, 1)
        self.criterion=nn.BCEWithLogitsLoss()
    def cal_loss(self,predict,gts):
        for i in range(len(predict)):
            if i==0:
                loss=self.criterion(predict[i].squeeze(), (gts[i]>0).squeeze().float())
            else:
                loss+=self.criterion(predict[i].squeeze(), (gts[i]>0).squeeze().float())
        return loss
    def forward(self, x,gts,get_mid=False):
        inp_size=x.shape[-2:]
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        edge=self.res4(x)*F.interpolate(self.attention4(d4), x.shape[-2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        edge = self.res3(self.conv4(edge)) *F.interpolate(self.attention3(d3), x.shape[-2:], mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        edge = self.res2(self.conv3(edge)) * F.interpolate(self.attention2(d2), x.shape[-2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(torch.cat([d2, x], 1))
        edge = self.res1(self.conv2(edge))*F.interpolate(self.attention1(d1), x.shape[-2:], mode='bilinear', align_corners=True)
        edge_result=F.interpolate(self.conv1(edge), inp_size, mode='bilinear', align_corners=True)
        f=self.psp(torch.cat([d1,F.interpolate(edge, inp_size, mode='bilinear', align_corners=True)],1))
        mid=f
        finalseg = self.finalconv(f)

        if get_mid:
            loss = self.cal_loss([finalseg, edge_result], [gts[0],gts[1]])
            return loss,[finalseg,edge_result,mid]
        else:
            loss=self.cal_loss([finalseg, edge_result], [gts[0], gts[1]])
            return loss, [finalseg,edge_result]
class SG_EPUNet(nn.Module):
    def __init__(self,
                 num_classes,
                 criterion,pretrain=False):
        super().__init__()
        self.criterion=criterion
        self.beforeUnet=EPUNet(1,criterion)
        self.afterUnet=EPUNet(1,criterion)
        self.disconv = nn.Sequential(nn.Conv2d(64*2, 64, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, num_classes, 1))
        self.bn1=nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bce=nn.BCEWithLogitsLoss(reduce=False)
    def cal_loss(self,predict,gts,trainset):
        [bpredict_seg, bpredict_edge, bmid], [apredict_seg, apredict_edge, amid], chg = predict
        [bmask,bedge],[amask,aedge]=gts
        amask=amask>0
        if self.training:
            tloss= self.bce(apredict_seg.squeeze(), amask.squeeze().float()).mean((1,2)) + self.bce(apredict_edge.squeeze(),aedge.squeeze().float()).mean((1,2))
            bmask = bmask.float()
            bmask[bmask == 0] = -1
            bmask[bmask > 0] = 1
            bmask = bmask * 5
            fseg=torch.sigmoid(chg.squeeze())*apredict_seg.squeeze()+\
                 bmask.squeeze()*(1-torch.sigmoid(chg.squeeze()))
            tloss+= self.bce(fseg, (amask>0).squeeze().float()).mean((1,2))
            vloss= self.bce(apredict_seg.squeeze(),(fseg>0).squeeze().float()).mean((1,2))
            floss=tloss*trainset.float()+vloss*(1-trainset.float())
        else:
            bmask = bmask.float()
            bmask[bmask == 0] = -1
            bmask[bmask > 0] = 1
            bmask = bmask * 5
            fseg = torch.sigmoid(chg.squeeze()) * apredict_seg.squeeze() + \
                   bmask.squeeze() * (1 - torch.sigmoid(chg.squeeze()))
            floss= self.bce(fseg, (amask > 0).squeeze().float()).mean((-1, -2))
        return floss,\
               [[torch.sigmoid(bpredict_seg),torch.sigmoid(bpredict_edge)],\
                [torch.sigmoid(apredict_seg),torch.sigmoid(apredict_edge)],\
                torch.sigmoid(chg),torch.sigmoid(fseg)]

    def forward(self, x,gts,trainset):
        _,pred1=self.beforeUnet(x[0],gts[0],True)
        pred1[-1]=pred1[-1].detach()
        loss2,pred2=self.afterUnet(x[1],gts[1],True)
        chg=self.disconv(torch.cat((pred1[-1]-pred2[-1],pred2[-1]-pred1[-1]),1))
        loss,pred=self.cal_loss([pred1,pred2,chg],gts,trainset)
        return loss2+loss,pred


