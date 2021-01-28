import matplotlib
matplotlib.use('tkagg')
import torch
import torch.optim as optim
import os
from core.utils.data_utils import directoryIterator
from core import configures
from osgeo import gdal, gdalconst
from core.utils.prepare_utils import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import torch.backends.cudnn as cudnn
from core.utils.sync_batchnorm import convert_model
from core.nets.SGEPUNet import EPUnet,SGEPUnet
from torch.nn.modules import loss
import numpy as np

cfg=configures.cfg
args=cfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion =loss.BCEWithLogitsLoss()
net=SGEPUnet(1,criterion)
writer = prep_experiment(args,net)
net=convert_model(net)
net=torch.nn.parallel.DataParallel(net.to(device))
train_loader,val_loader=directoryIterator.Loader(train_batch=args.batch_size,val_batch=args.batch_size,modelname=args.dataset,basedir=args.project_path,data_basedir=cfg.data_dir,cfg=cfg,\
train_txt=cfg.train_txt,val_txt=cfg.val_txt)
cudnn.benchmark=True
predict=torch.zeros((128*60,128*103))
stat=torch.zeros((128*60,128*103))
net.eval()
hist=0
with torch.no_grad():
    for iter_num, data in enumerate(val_loader):
        if iter_num%10==9:
            print(iter_num,'in',len(val_loader))
        (binputs, bmask, bedge), (ainputs, amask, aedge), img_names = data
        batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
        nrow=int(img_names[0].split('.')[0].split('_')[0])
        ncol = int(img_names[0].split('.')[0].split('_')[1])
        binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
        ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
        img = binputs.squeeze().permute((1, 2, 0)).cpu().numpy()
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        binputs = torch.tensor(img5).cuda()
        img = ainputs.squeeze().permute((1, 2, 0)).cpu().numpy()
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        ainputs = torch.tensor(img5).cuda()
        img = bmask.squeeze().cpu().numpy()
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        bmask = np.concatenate([img3, img4]).transpose(0, 1, 2)
        bmask = torch.tensor(bmask).cuda()
        loss,outputs = net([binputs,ainputs],[[bmask,bmask],[bmask,bmask]],True)
        mask = outputs[-1].squeeze().cpu().detach().numpy()
        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]
        outputs=torch.tensor(mask3/8)
        hist+=fast_hist((outputs>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),
                  2)
iou=hist[1][1]/(hist[1][1]+hist[0][1]+hist[1][0])
dataset = gdal.Open(os.path.join(cfg.data_dir,'after_test.tif'), gdalconst.GA_ReadOnly)
geo_transform = dataset.GetGeoTransform()
projection=dataset.GetProjection()
del dataset
afname=os.path.join(args.project_path,'predict',args.read_name.split('.')[0]+'_chg_'+str(iou)[2:6]+".tif")
target_ds = gdal.GetDriverByName('GTiff').Create(afname, xsize=stat.shape[1], ysize=stat.shape[0], bands=1,
                                                 eType=gdal.GDT_Byte)
target_ds.SetProjection(projection)
target_ds.SetGeoTransform((geo_transform[0],geo_transform[1]*2,geo_transform[2],\
                           geo_transform[3],geo_transform[4],geo_transform[5]*2))
ut_band = target_ds.GetRasterBand(1)
ut_band.WriteArray((((predict.float()/stat.float())).numpy()))
del target_ds