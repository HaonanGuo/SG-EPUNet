import torch
from torch.utils.data import Dataset,DataLoader
import csv
import os
import cv2
from PIL import Image
from .dataTransform import dataTransform
import numpy as np

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
        self.bimgFiles=[]
        self.blabFiles=[]
        self.aimgFiles=[]
        self.alabFiles=[]
        self.clabFiles = []
        self.ambiFiles=[]
        self.bmbiFiles=[]
        self.COLOR_MAP=color_map
        try:
            with open(txtdir) as f:
                Files = [tuple(line) for line in csv.reader(f)]
            Files = np.array(Files)

        except:
            with open(txtdir.replace('train','all').replace('test','all').replace('val','all')) as f:
                Files = [tuple(line) for line in csv.reader(f)]
            Files = np.array(Files)
        for i in range(len(Files)):
            if Files[i][0][-3:]!='png':
                fname=str(Files[i][0])+'.'+'png'
            else:
                fname=Files[i][0]
            if 'train' in txtdir:
                if int(fname.split('.')[0].split('_')[-1])<=100:
                    continue
            elif 'test' in txtdir:
                if int(fname.split('.')[0].split('_')[-1])>100:
                    continue
            elif 'all' in txtdir:
                pass
            else:
                raise NameError
            assert os.path.isfile(os.path.join(data_basedir, 'image', 'before', fname))
            assert os.path.isfile(os.path.join(data_basedir, 'label', 'before', fname))
            self.bimgFiles.append(os.path.join(data_basedir, 'image', 'before', fname))
            self.blabFiles.append(os.path.join(data_basedir, 'label', 'before', fname))
            self.aimgFiles.append(os.path.join(data_basedir, 'image', 'after', fname))
            self.alabFiles.append(os.path.join(data_basedir, 'label', 'after', fname))
        if os.path.exists(os.path.join(project_basedir, 'models', modelname + 'before' + '.txt')):
            f = open(os.path.join(project_basedir, 'models', modelname + 'before' + '.txt'), 'r')
            self.bmeans = np.array(f.readline().split(' '), dtype=np.float)
            self.bstdevs = np.array(f.readline().split(' '), dtype=np.float)
            f.close()
            f = open(os.path.join(project_basedir, 'models', modelname + 'after' + '.txt'), 'r')
            self.ameans = np.array(f.readline().split(' '), dtype=np.float)
            self.astdevs = np.array(f.readline().split(' '), dtype=np.float)
            f.close()
            cfg.ameans=self.ameans
            cfg.bmeans = self.bmeans
            cfg.astdevs=self.astdevs
            cfg.bstdevs = self.bstdevs
            return
        Files = np.array(Files)
        np.random.shuffle(Files)
        for i in range(len(Files)):
            if i > 500:
                break
            img = cv2.imread(os.path.join(data_basedir, 'image', Files[i][0]))
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
            f = open(os.path.join(project_basedir, 'models', modelname + '.txt'), 'w')
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
        return len(self.bimgFiles)

    def __getitem__(self, index):
        # print(index)

        before_image = np.array(Image.open(self.bimgFiles[index]).convert('RGB'))
        before_label = np.array(Image.open(self.blabFiles[index]).convert('L'))
        assert before_image.shape[0] == before_label.shape[0] and before_image.shape[1] == before_label.shape[1]
        label_temp = np.zeros((before_label.shape[0], before_label.shape[1]))
        for num, color_idx in enumerate(self.COLOR_MAP):
            label_temp += np.ones((before_label.shape[0], before_label.shape[1])) * (before_label == color_idx) * num
        before_label = label_temp
        after_image = np.array(Image.open(self.aimgFiles[index]).convert('RGB'))
        after_label = np.array(Image.open(self.alabFiles[index]).convert('L'))
        assert after_image.shape[0] ==after_label.shape[0] and after_image.shape[1] == after_label.shape[1]
        label_temp = np.zeros((after_label.shape[0], after_label.shape[1]))
        for num, color_idx in enumerate(self.COLOR_MAP):
            label_temp += np.ones((after_label.shape[0], after_label.shape[1])) * (after_label == color_idx) * num
        after_label = label_temp


        # self.transform.show_transform(before_image,before_label,after_image,after_label)
        if self.transform is not None:
            random_num=np.random.rand(20)
            # random_num=np.zeros(20)
            transformer= self.transform.get_random_transform(self.target_size[0],means=self.bmeans,stdevs=self.bstdevs)
            # t,change_label,t,t = self.transform.apply_transform(np.ones_like(after_image), change_label,np.ones_like(after_mbi),transformer,random_num=random_num,
            #                                                             training=self.training)
            before_image, before_label,before_edge = self.transform.apply_transform(before_image, before_label,transformer,random_num=random_num,
                                                                        training=self.training)
            transformer['mean']=self.ameans
            transformer['std'] = self.astdevs
            # temp,change_label,temp,temp = self.transform.apply_transform(np.ones_like(after_image),transformer,random_num=random_num,
            #                                                             training=self.training)
            after_image,after_label,after_edge = self.transform.apply_transform(after_image, after_label,transformer,random_num=random_num,
                                                                        training=self.training)

        return (before_image, before_label, before_edge),(after_image,after_label,after_edge), self.bimgFiles[index].split('/')[-1]

class mixDataset():
    def __init__(self,train_loader,val_loader):
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.tlen=len(train_loader.bimgFiles)
        self.vlen=len(val_loader.bimgFiles)
        self.iter=0
        self.stat=1
        self.load_mix=False
        self.training=False
    def __len__(self):
        return self.tlen+self.vlen
    def __getitem__(self, index):
        self.iter+=1
        if self.load_mix is False:
            if index>self.tlen:
                idx=int(index/self.__len__()*self.tlen)
                if idx>=self.tlen:
                    idx=idx-1
            else:
                idx=index
        else:
            idx = index
        if idx<=self.tlen:
            if idx==self.tlen:
                idx=idx-1
            return self.train_loader.__getitem__(idx),True
        else:
            return self.val_loader.__getitem__(idx-self.tlen),False
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
    if get_mix:
        mix_data=mixDataset(train_data,valid_data)
        train_loader = DataLoader(dataset=mix_data, batch_size=train_batch, shuffle=True,drop_last=True,num_workers=cfg.worker_num)
        valid_loader = DataLoader(dataset=valid_data, batch_size=val_batch,shuffle=True,drop_last=True,num_workers=cfg.worker_num)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=train_batch, shuffle=True,drop_last=True,num_workers=cfg.worker_num)
        valid_loader = DataLoader(dataset=valid_data, batch_size=val_batch,shuffle=False,drop_last=True,num_workers=cfg.worker_num)
    return train_loader,valid_loader
