import torch
from torch.utils.data import Dataset,DataLoader
import csv
import os
import cv2
from PIL import Image
from .dataTransform import dataTransform
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as trans
"""
Expext Directory Form:
-Folder
  -image
  -label
  train.txt
  val.txt
  
"""

class loadDataset(Dataset):
    def __init__(self,cfg,
                 project_basedir,
                 data_basedir,
                 txtdir,
                 color_map,
                 modelname=None,
                 target_size=None,
                 transform=None,
                 training=True):
        self.project_basedir=project_basedir,
        self.data_basedir=data_basedir,
        self.modelname=modelname
        self.transform = transform
        self.txtdir=txtdir,
        self.target_size = target_size,
        self.training=training
        self.imgFiles=[]
        self.labFiles=[]
        self.COLOR_MAP=color_map

        if training==True:
            self.mode='train'

        else:
            self.mode='val'
        lis = os.listdir(os.path.join('/data/haonan.guo/WHU_BUILDING',self.mode,'image'))
        for fname in lis:
            assert os.path.isfile(os.path.join('/data/haonan.guo/WHU_BUILDING',self.mode, 'image', fname))
            self.imgFiles.append(os.path.join('/data/haonan.guo/WHU_BUILDING',self.mode, 'image',  fname))
            self.labFiles.append(os.path.join('/data/haonan.guo/WHU_BUILDING',self.mode,  'label', fname))


        if os.path.exists(os.path.join(project_basedir, 'models', 'WHU_Building' + '.txt')):
            f = open(os.path.join(project_basedir, 'models', '18' + '.txt'), 'r')
            self.means = np.array(f.readline().split(' '), dtype=np.float)
            self.stdevs = np.array(f.readline().split(' '), dtype=np.float)
            f.close()
            cfg.ameans=self.means
            cfg.bmeans = self.means

            return
        Files = np.array(self.imgFiles)
        np.random.shuffle(Files)
        for i in range(len(Files)):
            if i > 500:
                break
            img = cv2.imread(Files[i] )
            if img.shape[0] != img.shape[1]:
                continue
            if i == 0:
                imgs = np.expand_dims(img, axis=-1)
            else:
                img = img[:, :, :, np.newaxis]
                imgs = np.concatenate((imgs, img), axis=3)
        imgs = imgs.astype(np.float32) / 255.
        self.means = []
        self.stdevs = []
        for i in range(3):
            pixels = imgs[:, :, i, :].ravel()  # 拉成一行
            self.means.append(np.mean(pixels))
            self.stdevs.append(np.std(pixels))
        self.means.reverse()  # BGR --> RGB
        self.stdevs.reverse()
        if modelname is not None:
            f = open(os.path.join(project_basedir, 'models', 'WHU_Building' + '.txt'), 'w')
            for i in range(len([self.means, self.stdevs])):
                s = str([self.means, self.stdevs][i]).replace('[', '').replace(']', '')
                s = s.replace("'", '').replace(',', '') + '\n'
                f.write(s)
            print('Mean:', self.means, '  Stds:', self.stdevs)
            f.close()
        else:
            print('Save model mean std dir not exist!')
            raise NameError

    def __len__(self):
        return len(self.imgFiles)

    def __getitem__(self, index):
        # print(index)
        image = np.array(Image.open(self.imgFiles[index]).convert('RGB'))
        label = np.array(Image.open(self.labFiles[index]).convert('L'))
        assert image.shape[0] == label.shape[0] and image.shape[1] == label.shape[1]
        if self.transform is not None:
            random_num=np.random.rand(20)
            transformer= self.transform.get_random_transform(self.target_size[0],means=self.means,stdevs=self.stdevs)
            image, label,edge = self.transform.apply_transform(image, label,transformer,random_num=random_num,
                                                                        training=self.training)

        return (image, label,edge), (image, label,edge),self.imgFiles[index].split('/')[-1]

def Loader(train_batch,val_batch,modelname,basedir,data_basedir,cfg,get_mix=False,train_txt='train.txt',val_txt='val.txt'):
    T = dataTransform((cfg.target_size[0], cfg.target_size[1]), cfg.target_size[2],
                      rotation_range=cfg.AUGMENTATION_CONFIG.rotation_range,
                      crop=cfg.crop,crop_size=cfg.crop_size,
                      height_shift_range=cfg.AUGMENTATION_CONFIG.width_shift_range, width_shift_range=cfg.AUGMENTATION_CONFIG.width_shift_range,
                      zoom_range=cfg.AUGMENTATION_CONFIG.zoom_range, zoom_maintain_shape=True,
                      horizontal_flip=cfg.AUGMENTATION_CONFIG.horizontal_flip, vertical_flip=cfg.AUGMENTATION_CONFIG.vertical_flip,
                      channel_shift_range=cfg.AUGMENTATION_CONFIG.channel_shift_range,
                      brightness_range=cfg.AUGMENTATION_CONFIG.channel_shift_range,
                      use_add_noise=cfg.AUGMENTATION_CONFIG.use_add_noise,use_blur=cfg.AUGMENTATION_CONFIG.use_blur,
                      use_gamma_transform=cfg.AUGMENTATION_CONFIG.use_gamma_transform,use_random_strech=cfg.AUGMENTATION_CONFIG.use_random_stretch,
                      hsv_strengthen=cfg.AUGMENTATION_CONFIG.hsv_strengthen)
    train_data = loadDataset(cfg,
             project_basedir=basedir,
             data_basedir=data_basedir,
             txtdir=os.path.join(data_basedir,train_txt),
             target_size=cfg.target_size,
            color_map=cfg.COLOR_MAP,
             training=True,
             transform=T,modelname=modelname)
    valid_data = loadDataset(cfg,
        project_basedir=basedir,
        data_basedir=data_basedir,
        txtdir=os.path.join(data_basedir, val_txt),
        target_size=cfg.target_size,
        color_map=cfg.COLOR_MAP,
        training=False,
        transform=T, modelname=modelname)

    train_loader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=True,drop_last=True,num_workers=cfg.worker_num)
    valid_loader = DataLoader(dataset=valid_data, batch_size=val_batch,shuffle=True,drop_last=True,num_workers=cfg.worker_num)
    return train_loader,valid_loader
