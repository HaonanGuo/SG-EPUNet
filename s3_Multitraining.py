import torch
import torch.optim as optim
import os
from core.utils.data_utils import directoryIterator
from core import sgepunet_configures
from datetime import datetime,timedelta
import torch.backends.cudnn as cudnn
from core.utils.sync_batchnorm import convert_model
from core.nets.Unets import SG_EPUNet
from torch.nn.modules import loss
import numpy as np
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
cfg=sgepunet_configures.cfg
args=cfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=SG_EPUNet(1,loss.BCEWithLogitsLoss())
net.load_state_dict(torch.load( os.path.join(cfg.project_path,'models',cfg.read_name)))
net=convert_model(net)
optimizer =optim.Adam(net.parameters(),lr=args.base_lr)
scheduler=optim.lr_scheduler.StepLR(optimizer,8,0.1)
net=torch.nn.parallel.DataParallel(net.to(device))
train_loader,val_loader=directoryIterator.Loader(train_batch=args.batch_size,val_batch=args.batch_size,modelname=args.dataset,basedir=args.project_path,data_basedir=cfg.data_dir,cfg=cfg,\
train_txt=cfg.train_txt,val_txt=cfg.val_txt,get_mix=True)
cudnn.benchmark=True
ioustat=[]
valloss=9999
val_best_loss=9999
if os.path.exists(os.path.join(cfg.project_path, 'models', cfg.read_name)):
    print('Pre-Validating....')
    loss_sigma=0
    net.eval()
    hist=0
    with torch.no_grad():
        for iter_num, data in enumerate(val_loader):
            (binputs, bmask, bedge), (ainputs, amask, aedge), img_names= data
            batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
            binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
            ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
            mask=bmask.cuda()
            inputs = [binputs, ainputs]
            gts = [[bmask, bedge], [amask, aedge]]
            loss,outputs = net(inputs,gts,True)
            [bseg_predict,bedge_predict],[aseg_predict,aedge_predict],chg,fseg=outputs
            hist+=fast_hist((fseg>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),2)
            loss_sigma += float(loss.mean().item())

    iou=(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1]
    valloss=1-iou
    val_best_loss=valloss
    print('Best Loss:'+ str(val_best_loss))
    print('IOU:',(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1])
for epoch in range(cfg.epoch):
    loss_sigma = 0.0
    segcorrect=0.0
    segtotal=0.0
    stp_dtime=timedelta(0,0,0)
    eph_dtime=datetime.now()
    stp_stime = datetime.now()
    warmup_epochs=cfg.warmup_epochs
    if epoch==5:
        print('Start Loading Valset!')
        train_loader.dataset.load_mix=True
    if warmup_epochs and (epoch+1) < warmup_epochs:
        warmup_percent_done = (epoch+1) / warmup_epochs
        warmup_learning_rate = cfg.base_lr * warmup_percent_done  # gradual warmup_lr
        learning_rate = warmup_learning_rate
        optimizer.param_groups[0]['lr']=learning_rate
    elif  warmup_epochs and (epoch+1)==warmup_epochs:
        optimizer.param_groups[0]['lr']=cfg.base_lr
        torch.save(net.module.state_dict(), os.path.join(cfg.project_path,'models',cfg.save_name))
    net.train()
    net.module.beforeUnet.eval()
    net.module.beforeUnet.requires_grad_(False)
    print('Epoch:[{:0>3}/{:0>3}] '.format(epoch + 1, cfg['epoch'])+'Learning rate '+str(optimizer.param_groups[0]['lr']))
    data_loader=train_loader
    for iter_num, data in enumerate(data_loader):
        (binputs, bmask, bedge), (ainputs, amask, aedge), img_names= data[0]
        trainval=data[1]
        batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
        binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
        ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
        optimizer.zero_grad()
        inputs = [binputs, ainputs]
        gts = [[bmask, bedge], [amask, aedge]]
        loss, outputs = net(inputs, gts,trainval)
        _,_,_,fseg = outputs
        loss.mean().backward()
        optimizer.step()
        with torch.no_grad():
            seg_predicted = fseg
            loss_sigma += float(loss.mean().item())
            segtotal+=bmask.shape.numel()
            segcorrect += ((seg_predicted>0.5).squeeze().long().cpu() == (
                    amask>0).long().cpu().squeeze()).cpu().squeeze().sum().numpy()
        if iter_num % 10 == 9:
            stp_dtime+=datetime.now()-stp_stime
            all_time=stp_dtime.seconds/iter_num*(len(data_loader))
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch:[{:0>3}/{:0>3}] Iteration:[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} Time:{:0>2}:{:0>2}/{:0>2}:{:0>2}".format\
                    (
                epoch + 1, cfg['epoch'], iter_num + 1, len(data_loader), loss_avg, segcorrect/segtotal,\
                    int(stp_dtime.seconds//60),int(stp_dtime.seconds%60),int(all_time//60),int(all_time%60)))
            stp_stime=datetime.now()
    if (epoch+1) < warmup_epochs:
        print('In pre-hotting status!Saving model!')
        torch.save(net.module.state_dict(),  os.path.join(cfg.project_path,'models',cfg.save_name))
        continue
    scheduler.step()
    if epoch%3==2 or epoch==0 or epoch==warmup_epochs-1:
        print('Validating...')
    else:
        continue
    net.eval()
    hist=0
    with torch.no_grad():
        for iter_num, data in enumerate(val_loader):
            if iter_num%10==9:
                print('Validaing: Iteration:[{:0>3}/{:0>3}]'.format(iter_num+1,len(val_loader)))
            (binputs, bmask, bedge), (ainputs, amask, aedge), img_names = data
            batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
            binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
            ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
            inputs = [binputs, ainputs]
            gts = [[bmask, bedge], [amask, aedge]]
            _, outputs = net(inputs, gts,True)
            _,_,_, fseg = outputs
            hist+=fast_hist((fseg>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),2)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    print("Validating: Epoch:[{:0>3}/{:0>3}] Time:{:0>2}:{:0>2} Validation {}:{:.4f}".format\
        (epoch + 1, cfg['epoch'],int((datetime.now()-eph_dtime).seconds//60),int((datetime.now()-eph_dtime).seconds%60),'IOU:',iou[-1]))
    print('IOU:',(np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))[-1])
    valloss=1-iou[-1]
    if valloss < val_best_loss:
        val_best_loss = valloss
        torch.save(net.module.state_dict(),  os.path.join(cfg.project_path,'models',cfg.save_name))
print('Finished Training')