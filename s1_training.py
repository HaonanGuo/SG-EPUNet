import torch
import torch.optim as optim
import os
from core.utils.data_utils import directoryIterator
from core import epunet_configures
from datetime import datetime,timedelta
import torch.backends.cudnn as cudnn
from core.utils.sync_batchnorm import convert_model
from core.nets.Unets import EPUNet
from torch.nn.modules import loss
import numpy as np
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
cfg=epunet_configures.cfg
args=cfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion =loss.BCEWithLogitsLoss()
net=EPUNet(1,criterion)
net=convert_model(net)
optimizer =optim.Adam(net.parameters(),lr=args.base_lr,weight_decay=5e-4)
scheduler=optim.lr_scheduler.StepLR(optimizer,15,0.1)
net=torch.nn.parallel.DataParallel(net.to(device))
train_loader,val_loader=directoryIterator.Loader(train_batch=args.batch_size,val_batch=args.batch_size,modelname=args.dataset,basedir=args.project_path,data_basedir=cfg.data_dir,cfg=cfg,\
train_txt=cfg.train_txt,val_txt=cfg.val_txt)
cudnn.benchmark=True
for epoch in range(cfg.epoch):
    loss_sigma = 0.0
    segcorrect=0.0
    segtotal=0.0
    stp_dtime=timedelta(0,0,0)
    eph_dtime=datetime.now()
    stp_stime = datetime.now()
    warmup_epochs=cfg.warmup_epochs
    if warmup_epochs and (epoch+1) < warmup_epochs:
        warmup_percent_done = (epoch+1) / warmup_epochs
        warmup_learning_rate = cfg.base_lr * warmup_percent_done  # gradual warmup_lr
        learning_rate = warmup_learning_rate
        optimizer.param_groups[0]['lr']=learning_rate
    elif  warmup_epochs and (epoch+1)==warmup_epochs:
        optimizer.param_groups[0]['lr']=cfg.base_lr
        torch.save(net.module.state_dict(), os.path.join(cfg.project_path,'models',cfg.save_name))

    net.train()
    print('Epoch:[{:0>3}/{:0>3}] '.format(epoch + 1, cfg['epoch'])+'Learning rate '+str(optimizer.param_groups[0]['lr']))
    for iter_num, data in enumerate(train_loader):
        (binputs, bmask, bedge), _, img_names = data
        batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
        binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
        optimizer.zero_grad()
        image = binputs
        label = bmask
        edge = bedge
        loss, outputs = net(image, [label, edge])
        outputs = outputs[0]
        loss.mean().backward()
        optimizer.step()
        with torch.no_grad():
            seg_predicted = torch.sigmoid(outputs)
            loss_sigma += float(loss.mean().item())
            segtotal+=label.shape.numel()
            segcorrect += ((seg_predicted>0.5).squeeze().long().cpu() == (
                    label>0).long().cpu().squeeze()).cpu().squeeze().sum().numpy()
        if iter_num % 10 == 9:
            stp_dtime+=datetime.now()-stp_stime
            all_time=stp_dtime.seconds/iter_num*(len(train_loader))
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch:[{:0>3}/{:0>3}] Iteration:[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} Time:{:0>2}:{:0>2}/{:0>2}:{:0>2}".format\
                    (
                epoch + 1, cfg['epoch'], iter_num + 1, len(train_loader), loss_avg, segcorrect/segtotal,\
                    int(stp_dtime.seconds//60),int(stp_dtime.seconds%60),int(all_time//60),int(all_time%60)))
            stp_stime=datetime.now()
    print('Saving model!')
    torch.save(net.module.state_dict(),  os.path.join(cfg.project_path,'models',cfg.save_name))
    scheduler.step()
    net.eval()
    loss_sigma = 0.0
    cls_num = 2
    hist=0
    if epoch%5==0:
        with torch.no_grad():
            for iter_num, data in enumerate(val_loader):
                if iter_num%10==9:
                    print('Validaing: Iteration:[{:0>3}/{:0>3}]'.format(iter_num+1,len(val_loader)))
                (binputs, bmask, bedge), (ainputs, amask, aedge), img_names = data
                batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
                binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
                ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
                image=ainputs
                label=amask
                edge=aedge
                loss,outputs = net(image,[label,edge])
                outputs=outputs[0]
                loss_sigma += float(loss.mean().item())
                seg_prediction = (torch.sigmoid(outputs)>0.5).cpu().detach()
                hist+=fast_hist(seg_prediction.numpy().flatten(), label.cpu().numpy().flatten(),
                                       args.n_class)
        acc = np.diag(hist).sum() / hist.sum()
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        print("Validating: Epoch:[{:0>3}/{:0>3}] Time:{:0>2}:{:0>2} Validation Loss:{:.4f} Validation {}:{:.4f}".format\
                        (
                    epoch + 1, cfg['epoch'],int((datetime.now()-eph_dtime).seconds//60),int((datetime.now()-eph_dtime).seconds%60),
                loss_sigma / len(val_loader),\
                'IOU:',iou[-1]))
print('Finished Training')