import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
base_dir='/home/guest/ghn/WHU_Building'
a='after'
b='before'
c='change'
i='image'
l='label'
lis=os.listdir('/home/guest/ghn/WHU_Building/label/before')
for id in tqdm(lis):
    before_image = Image.open(os.path.join(base_dir,i,b,id)).convert('RGB')
    before_label = Image.open(os.path.join(base_dir,l,b,id)).convert('L')
    before_image = np.array(before_image)
    before_label = np.array(before_label)
    after_image = Image.open(os.path.join(base_dir,i,a,id)).convert('RGB')
    after_label = Image.open(os.path.join(base_dir,l,a,id)).convert('L')
    after_image = np.array(after_image)
    after_label = np.array(after_label)
    change_label = Image.open(os.path.join(base_dir, l,c, id)).convert('L')
    change_label = np.array(change_label)
    if before_image.shape[0:2]!=(512,512):
        temp = np.zeros((512, 512, 3))
        temp[:before_image.shape[0], :before_image.shape[1]] = before_image
        t = Image.fromarray(temp.astype(np.uint8))
        t.save(os.path.join(base_dir,i,b,id))
        print('before image:',id)
    if after_image.shape[0:2]!=(512,512):
        temp = np.zeros((512, 512, 3))
        temp[:after_image.shape[0], :after_image.shape[1]] = after_image
        t = Image.fromarray(temp.astype(np.uint8))
        t.save(os.path.join(base_dir,i,a,id))
        print('after image',id)
    if before_label.shape[0:2]!=(512,512):
        print('before label:',id)
    if after_label.shape[0:2]!=(512,512):
        print('after label',id)