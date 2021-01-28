import torch
import torch.nn as nn
import math
import numpy as np
import torchvision.transforms as trans
import torchvision.transforms.functional as F
from PIL import Image, ImageChops
import cv2
from PIL import ImageEnhance
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor,normalize
import sys
sys.path.append(r'D:\SemanticSeg_Pytorch')
from core import configures as cfg
import core.utils.edge_utils as edge_utils
"""
Implementation of image Augmentation, including:
    (1) Flip                            √
    (2)Crop：
        RandomCrop                      √
        CenterCrop                      √
    (3) Rotate                          √
    (4) Shift                           √
    (5) Add Noise                       √
    (6) Blur                            √
    (7) Stretch                        
         Linear                         √   
         Gamma                          √
"""
class dataTransform():
    def __init__(self,target_size,channel,channel_axis=2,nclass=2,
                 crop=None,
                 crop_size=None,
                 rotation_range=0.,
                 height_shift_range=0.,width_shift_range=.0,
                 zoom_range=[1.,1.],zoom_maintain_shape=True,
                horizontal_flip=False,vertical_flip=False,
                channel_shift_range=0.,
                brightness_range=[1.,1.],
                use_random_strech=False,use_add_noise=False,
                 use_blur=False,use_gamma_transform=False,
                 hsv_strengthen=False
                 ):
        self.mean=[0.5,0.5,0.5]
        self.std=[0.25,0.25,0.25]
        self.channel= channel
        self.channel_axis=channel_axis
        self.target_h=target_size[0]
        self.target_w=target_size[1]
        self.nclass=nclass
        self.crop_mode=crop
        self.crop_size=crop_size
        self.rotation_range=rotation_range
        self.height_shift_range=height_shift_range
        self.width_shift_range=width_shift_range
        self.zoom_range=zoom_range
        self.zoom_maintain_shape=zoom_maintain_shape
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.channel_shift_range=channel_shift_range
        self.brightness_range=brightness_range
        self.use_random_strech = use_random_strech
        self.use_add_noise = use_add_noise,
        self.use_blur = use_blur,
        self.use_gamma_transform = use_gamma_transform
        self.hsv_strengthen=hsv_strengthen
    def image_flip(self,image, left_right_axis=True):
        """ Flip the src and label images
        :param image: numpy array
        :param label: numpy array
        :param left_right_axis: True / False
        :return: the processed numpy arrays
        """
        axis = 1 if left_right_axis == True else 0
        image = np.flip(image, axis=axis)
        return image

    def image_randomcrop(self,image,label, crop_height, crop_width):
        """ Random Crop the src and label images
        :param image: numpy array
        :param label: numpy array
        :param crop_height: target height
        :param crop_width: target width
        :return: the processed numpy arrays
        """
        assert image.shape[1] >= crop_width and image.shape[0] >= crop_height
        image_width = image.shape[1]
        image_height = image.shape[0]
        x = np.random.randint(0, image_width - crop_width + 1)
        y = np.random.randint(0, image_height - crop_height + 1)
        return image[y:y + crop_height, x:x + crop_width],label[y:y + crop_height, x:x + crop_width]

    def image_randomresizecrop(self,image,size,scale=(0.4, 1), ratio=(1.33, 0.75), interpolation=2):
        def get_para():
            for attempt in range(10):
                area = image.size[0] * image.size[1]
                target_area = np.random.uniform(*scale) * area
                log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
                aspect_ratio = math.exp(np.random.uniform(*log_ratio))

                w = int(round(math.sqrt(target_area * aspect_ratio)))
                h = int(round(math.sqrt(target_area / aspect_ratio)))
                if w <= image.size[0] and h <= image.size[1]:
                    i = np.random.randint(0, image.size[1] - h)
                    j = np.random.randint(0, image.size[0] - w)
                    return i, j, h, w
        i, j, h, w = self.get_para(image, self.scale, self.ratio)
        return F.resized_crop(image, i, j, h, w,size, interpolation)

    def image_centercrop(self,image, crop_height, crop_width):
        centerh, centerw = image.shape[0] // 2, image.shape[1] // 2
        lh, lw = crop_height // 2, crop_width // 2
        rh, rw = crop_height - lh, crop_width - lw

        h_start, h_end = centerh - lh, centerh + rh
        w_start, w_end = centerw - lw, centerw + rw

        return image[h_start:h_end, w_start:w_end]

    def rotate(self,image, angle):
        M_rotate = cv2.getRotationMatrix2D((self.target_w / 2, self.target_h/ 2), angle, 1)
        xb = cv2.warpAffine(image, M_rotate, (self.target_w, self.target_h))
        return xb

    def image_shift(self,image, xoff, yoff, image_background_value=(0, 0, 0), label_background_value=0):
        """ applying shifting on the x-axis and the y-axis
        :param image: numpy array
        :param label: numpy array
        :param image_background_value: e.g, (0,0,0)
        :param label_background_value: e.g, 0
        :return: the processed numpy arrays
        """
        def offset(img, xoff, yoff, background_value):
            xoff=int(xoff)
            yoff=int(yoff)
            try:
                img=Image.fromarray(img)
            except:
                img=Image.fromarray(np.stack([img[:,:,0],img[:,:,0],img[:,:,0]],axis=-1).astype(np.uint8))
            c = ImageChops.offset(img, int(xoff), int(yoff))
            if int(xoff)>0:
                c.paste(background_value, (0, 0, xoff, img.size[1]))
            else:
                c.paste(background_value, (img.size[0]+xoff,0,img.size[0],img.size[1]))
            if int(yoff)>0:
                c.paste(background_value, (0, 0, img.size[0], yoff))
            else:
                c.paste(background_value, (0, img.size[1]+yoff, img.size[0], img.size[1]))
            return np.array(c, dtype=np.uint8)
        return offset(image, xoff, yoff, image_background_value)

    def image_scale(self,image, zoom_x, zoom_y, image_background_value=(0, 0, 0), label_background_value=0):
        image = Image.fromarray(image)
        assert zoom_x == zoom_y
        if zoom_x > 1:
            # the scaled image is larger than the original one
            # select subregion equal to the original size from the scaled image
            resized_size = (int(image.size[0] * zoom_x), int(image.size[1] * zoom_y))
            image_resize = image.resize(resized_size)
            margin_width = (resized_size[0] - image.size[0]) // 2
            margin_height = (resized_size[1] - image.size[1]) // 2

            image_result = image_resize.transform(image.size, Image.EXTENT, (
            margin_width, margin_height, margin_width + image.size[0], margin_height + image.size[1]))
            return np.array(image_result, dtype=np.uint8)
        else:
            # the scaled image is smaller than the original one
            # mapping the center part of the empty image with the scaled image
            resized_size = (int(image.size[0] * zoom_x), int(image.size[1] * zoom_y))
            image_resize = image.resize(resized_size)

            image_result = np.ones((image.size[1], image.size[0], len(image_background_value)), dtype=np.uint8) * \
                           image_background_value[0]
            margin_width = (image.size[0] - resized_size[0]) // 2
            margin_height = (image.size[1] - resized_size[1]) // 2
            image_result[margin_height:margin_height + resized_size[1], margin_width:margin_width + resized_size[0],
            :] = image_resize
            return np.array(image_result)

    def apply_brightness_shift(self,image,brightness):
        """Performs a brightness shift.
        # Arguments
            x: Input tensor. Must be 3D.
            brightness: Float. The new brightness value.
            channel_axis: Index of axis for channels in the input tensor.

        # Returns
            Numpy image tensor.

        # Raises
            ValueError if `brightness_range` isn't a tuple.
        """

        image=Image.fromarray(image.astype(np.uint8))
        imgenhancer_Brightness = ImageEnhance.Brightness(image)
        image = imgenhancer_Brightness.enhance(brightness)
        return image

    def random_gamma_transform(self,img, gamma_vari=1.2):
        img=np.array(img)
        gamma = np.random.uniform(1, gamma_vari)
        if np.random.random()<0.5:
            gamma=1.0/gamma
        return (255*np.power(img/255.0, gamma)).astype('int'),label

    def random_strech(self,img,label):
        # img=np.array(img)
        # label=np.array(label)
        # min_value, max_value=np.min(img),np.max(img)
        # b=np.random.randint(0,self.channel)
        # ratio=1+np.random.choice([-1,1])*np.random.random()/4
        # img[:,:,b]=ratio*img[:,:,b]
        # image = np.clip(img, min_value, max_value)
        return img,label

    def image_addnoise(self,image,factor=5, min_value=0, max_value=255):
        """ add noise to the src image
        :param image: numpy array
        :param label: numpy array
        :param factor: the sigma of the guess noise added the src image
        :return: the processed images
        """
        add_type=np.random.choice([1,2])
        image=np.array(image)
        t=(image>0)
        if add_type==1:
            noise = np.random.normal(loc=0.0, scale=1.0) * factor
            image = np.clip(image + noise, 0, 255)
        elif add_type==2:
            for i in range(np.random.randint(image.shape[1],image.shape[1]*2)):
                temp_x = np.random.randint(0, image.shape[0])
                temp_y = np.random.randint(0, image.shape[1])
                image[temp_x,temp_y,:] = 255
        return np.multiply(image,t).astype('int')

    def hsv_strength(self,image, hue_shift_limit=(-20, 20),
                                 sat_shift_limit=(-30, 30),
                                 val_shift_limit=(-30, 30)):
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    def blur(self,img):
        img=np.array(img)
        img = cv2.blur(img, (3, 3))
        return img

    def get_random_transform(self, img_shape,means,stdevs):
        """Generates random parameters for a transformation.
        # Arguments
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        if self.rotation_range:
            # angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            angle=np.random.choice([90,180,270])
        else:
            angle = 0

        if self.height_shift_range:
            try:
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except:
                tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
            if self.height_shift_range[0] < 2:
                tx *= img_shape[0]
        else:
            tx = 0

        if self.width_shift_range[0]:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if self.width_shift_range[0]< 2:
                ty *= img_shape[1]
        else:
            ty = 0


        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0][0], self.zoom_range[0][1], 2)
        if self.zoom_maintain_shape:
            zy = zx

        flip_horizontal = self.horizontal_flip
        flip_vertical =  self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range[0], self.channel_shift_range[0])

        brightness = None
        if self.brightness_range[0] != 0:
            brightness = np.random.uniform(self.brightness_range[0][0], self.brightness_range[0][1])

        transform_parameters = {'angle': angle,
                                'tx': tx,
                                'ty': ty,
                                'zx': zx,
                                'zy': zy,
                                'mean':means,
                                'std':stdevs,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                "use_random_strech":self.use_random_strech,
                                "use_add_noise":self.use_add_noise,
                                "use_blur":self.use_blur,
                                "use_gamma_transform":self.use_gamma_transform,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness,
                                'hsv_strengthen':self.hsv_strengthen}

        return transform_parameters

    def normalize(tensor, mean, std, inplace=False):
        """Normalize a tensor image with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not inplace:
            tensor = tensor.clone()

        mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor

    def apply_transform(self, image, label, trans_para,random_num, p=0.25,training=True,feed_one_hot=False):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            y: 2D tensor, single label
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.

        # Returns
            A transformed version of the input (same shape).
        """
        # random_num=np.random.rand(20)#0.22 Loss
        # random_num=np.ones(20)#0.27 Loss
        # random_num=np.zeros(20)
        cp=image
        image=np.array(image)
        label=np.array(label)

        if label.ndim == 2:
            label = np.expand_dims(label, -1)

        #Rotation
        if trans_para['angle'] != 0 and random_num[0]<p and training==True:
            image= self.rotate(image, trans_para['angle'] )
            label= self.rotate(label, trans_para['angle'] )
        if label.ndim == 2:
            label = np.expand_dims(label, -1)

        #Shift
        if (trans_para['tx'] != 0 or trans_para['ty'] != 0) and random_num[1]<p and training==True:
            image = self.image_shift(image, trans_para['tx'], trans_para['ty'] )
            label = self.image_shift(label, trans_para['tx'], trans_para['ty'])
        if label.ndim == 3:
            label =label[:,:,0]
            t=np.zeros((label.shape[0],label.shape[1],3))
            t[:,:,0]=label
            label=t.astype(np.uint8)
        #Scale
        if (trans_para['zx'] != 1 or trans_para['zy'] != 1) and random_num[2]<p and training==True:
            image = self.image_scale(image,trans_para['zx'], trans_para['zy'])
            label = self.image_scale(label, trans_para['zx'], trans_para['zy'])
        if label.ndim == 3:
            label = label[:, :, 0]

        #Chaanel_Shift

        #Flip
        if trans_para['flip_horizontal']!=False and training==True and random_num[4]<p:
            image = self.image_flip(image,True)
            label = self.image_flip(label, True)
        if trans_para['flip_vertical']!=False and training==True and random_num[5]<p:
            image= self.image_flip(image,False)
            label = self.image_flip(label, False)
        #Brigtness Adjust
        if trans_para['brightness'] is not None and random_num[6]<p and training==True:
            image,label = self.apply_brightness_shift(image,trans_para['brightness'])

        if trans_para["use_random_strech"]==True and random_num[7]<p and training==True:
            image,label = self.random_strech(image)
        if trans_para["use_add_noise"][0]==True and random_num[8]<p and training==True:
            image= self.image_addnoise(image)
        if trans_para["use_blur"][0]==True and random_num[9]<p and training==True:
            image= self.blur(image)
        if trans_para["use_gamma_transform"]==True and random_num[10]<p and training==True:
            image, label = self.random_gamma_transform(image)
        if trans_para["hsv_strengthen"]==True and random_num[10]<p and training==True:
            image=self.hsv_strength(image)
            """
        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(label)
        plt.show() 
            """


        try:
            image=Image.fromarray(image)
            image = image.resize((self.target_h,self.target_w),Image.LANCZOS)
        except:
            image=np.array(image)
            image = cv2.resize(image,(self.target_h,self.target_w),interpolation=cv2.INTER_LANCZOS4)
        finally:
            label = Image.fromarray(label)
            label = label.resize((self.target_h,self.target_w),Image.LANCZOS)
            label=np.array(label)
            image=np.array(image)
            if self.crop_mode:
                image,label=self.image_randomcrop(image,label,self.crop_size[0],self.crop_size[1])
        image=to_tensor(np.ascontiguousarray(image, dtype=np.float32))/255
        image=normalize(image,trans_para["mean"],trans_para["std"])
        label=to_tensor(np.ascontiguousarray(label, dtype=np.long))
        _edgemap = edge_utils.mask_to_onehot(np.array(label.squeeze()), 2)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
        edgemap = torch.from_numpy(_edgemap).float()
        return image,label.squeeze(),edgemap

    def apply_channel_shift(self,x,intensity=0.0, channel_axis=2):
        """Performs a channel shift.

        # Arguments
            x: Input tensor. Must be 3D.
            intensity: Transformation intensity.
            channel_axis: Index of axis for channels in the input tensor.

        # Returns
            Numpy image tensor.

        """
        if intensity is None:
            intensity=0
        if channel_axis==0:
            channel=x.shape[0]
        else:
            channel=x.shape[-1]
        x = np.rollaxis(x, channel_axis, 0)
        min_x, max_x = np.min(x), np.max(x)
        if channel_axis==0:
            channel_images = [
                np.clip(x[i,:,:] + intensity,min_x,max_x) for i in range(channel)]
        else:
            channel_images = [
                np.clip(x[i,:,:] + intensity*(max_x-min_x),min_x,max_x) for i in range(channel)]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x.astype(np.uint8)

    def show_transform(self,bimage,blabel,aimage,alabel):
        _edgemap = edge_utils.mask_to_onehot(np.array(blabel.squeeze()), 2)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
        bedge = _edgemap[0]
        _edgemap = edge_utils.mask_to_onehot(np.array(alabel.squeeze()), 2)
        _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, 2)
        aedge = _edgemap[0]
        self.show( bimage, blabel, bedge, aimage, alabel, aedge)
        plt.savefig('origin.png', dpi=300)
        self.show(self.rotate(bimage,-30),self.rotate(blabel,-30),self.rotate(bedge,30),
                  self.rotate(aimage,-30),self.rotate(alabel,-30),self.rotate(aedge,-30))
        plt.savefig('rotate.png', dpi=300)
        self.show(self.image_shift(bimage,-38,38),self.image_shift(np.expand_dims(blabel, -1),-38,38)[:,:,0],self.image_shift(np.expand_dims(bedge, -1),-38,38)[:,:,0],
                  self.image_shift(aimage, -38, 38),self.image_shift(np.expand_dims(alabel, -1),-38,38)[:,:,0],self.image_shift(np.expand_dims(aedge, -1),-38,38)[:,:,0])
        plt.savefig('shift.png', dpi=300)
        def expand(label):
            t = np.zeros((label.shape[0], label.shape[1], 3))
            t[:, :, 0] = label
            return t.astype(np.uint8)
        self.show(self.image_scale(bimage,0.85,0.85),self.image_scale(expand(blabel),0.85,0.85)[:,:,0],self.image_scale(expand(bedge),0.85,0.85)[:,:,0],
                  self.image_scale(aimage, 0.85, 0.85), self.image_scale(expand(alabel), 0.85, 0.85)[:,:,0],
                  self.image_scale(expand(aedge), 0.85, 0.85)[:,:,0])
        plt.savefig('zoom.png', dpi=300)
        self.show(self.image_flip(bimage,False),self.image_flip(blabel,False),self.image_flip(bedge,False),
                  self.image_flip(aimage,False), self.image_flip(alabel,False),  self.image_flip(aedge,False))
        plt.savefig('flip.png',dpi=300)
        return
    def show(self,bimage,blabel,bedge,aimage,alabel,aedge):
        plt.figure()
        plt.subplot(161)
        plt.imshow(bimage)
        plt.axis('off')
        plt.subplot(162)
        plt.imshow(blabel)
        plt.axis('off')
        plt.subplot(163)
        plt.imshow(bedge)
        plt.axis('off')
        plt.subplot(164)
        plt.imshow(aimage)
        plt.axis('off')
        plt.subplot(165)
        plt.imshow(alabel)
        plt.axis('off')
        plt.subplot(166)
        plt.imshow(aedge)
        plt.axis('off')
