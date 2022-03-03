import torch
import os
from core.utils.data_utils import directoryIterator
from core import sgepunet_configures as configures
from osgeo import gdal, gdalconst
import torch.backends.cudnn as cudnn
from core.utils.sync_batchnorm import convert_model
from core.nets.Unets import SG_EPUNet
from torch.nn.modules import loss
import numpy as np

cfg=configures.cfg
args=cfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion =loss.BCEWithLogitsLoss()
net=SG_EPUNet(1,criterion)
net.load_state_dict(torch.load(os.path.join(cfg.project_path, 'models', cfg.save_name)))
net=convert_model(net)
net=torch.nn.parallel.DataParallel(net.to(device))
_,val_loader=directoryIterator.Loader(train_batch=args.batch_size,val_batch=1,modelname=args.dataset,basedir=args.project_path,data_basedir=cfg.data_dir,cfg=cfg,\
train_txt=cfg.train_txt,val_txt=cfg.val_txt)
cudnn.benchmark=True
predict=torch.zeros((128*60,128*103))
stat=torch.zeros((128*60,128*103))
net.eval()
hist=0
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
with torch.no_grad():
    for iter_num, data in enumerate(val_loader):
        if iter_num%10==9:
            print(iter_num,'in',len(val_loader))
        (binputs, bmask, bedge), (ainputs, amask, aedge), img_names = data
        loss,outputs = net([binputs,ainputs],[[bmask,bmask],[bmask,bmask]],True)
        mask = outputs[-1].squeeze().cpu().detach().numpy()
        outputs=torch.tensor(mask)
        hist+=fast_hist((outputs>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),2)
        nrow=int(img_names[0].split('.')[0].split('_')[0])
        ncol = int(img_names[0].split('.')[0].split('_')[1])
        predict[nrow*128:nrow*128+256,ncol*128:ncol*128+256]+=outputs.cpu().detach().squeeze()
        stat[nrow*128:nrow*128+256,ncol*128:ncol*128+256]+=1

iou=hist[1][1]/(hist[1][1]+hist[0][1]+hist[1][0])
print(iou)
dataset = gdal.Open(os.path.join(cfg.data_dir,'after_test.tif'), gdalconst.GA_ReadOnly)
geo_transform = dataset.GetGeoTransform()
projection=dataset.GetProjection()
del dataset
afname=os.path.join(args.project_path,args.save_name.split('.')[0]+'_chg_'+str(iou)[2:6]+".tif")
target_ds = gdal.GetDriverByName('GTiff').Create(afname, xsize=stat.shape[1], ysize=stat.shape[0], bands=1,
                                                 eType=gdal.GDT_Byte)
target_ds.SetProjection(projection)
target_ds.SetGeoTransform((geo_transform[0],geo_transform[1]*2,geo_transform[2],\
                           geo_transform[3],geo_transform[4],geo_transform[5]*2))
ut_band = target_ds.GetRasterBand(1)
stat[stat==0]=1
pred=((predict.float()/stat.float())>0.5).int().numpy()
ut_band.WriteArray(pred*255)
del target_ds