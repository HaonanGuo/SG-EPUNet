import matplotlib
matplotlib.use('tkagg')
import torch
import torch.optim as optim
import os
from core.utils.data_utils import directoryIterator
from core import configures
from datetime import datetime,timedelta
from core.utils.prepare_utils import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import torch.backends.cudnn as cudnn
from core.utils.sync_batchnorm import convert_model
from core.nets.SGEPUNet import SGEPUnet
from torch.nn.modules import loss
import numpy as np
cfg=configures.cfg
args=cfg
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=SGEPUnet(1,loss.BCEWithLogitsLoss())
writer = prep_experiment(args,net)
net=convert_model(net)
optimizer =optim.Adam(net.parameters(),lr=args.base_lr,weight_decay=5e-4)
scheduler=optim.lr_scheduler.StepLR(optimizer,20,0.1)
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
    fhist=0
    ahist=0
    bhist=0
    with torch.no_grad():
        for iter_num, data in enumerate(val_loader):
            (binputs, bmask, bedge), (ainputs, amask, aedge), img_names= data
            batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
            binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
            ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
            # inputs=ainputs
            mask=bmask.cuda()
            inputs = [binputs, ainputs]
            gts = [[bmask, bedge], [amask, aedge]]
            loss,outputs = net(inputs,gts,True)
            [bseg_predict,bedge_predict],[aseg_predict,aedge_predict],chg,fseg=outputs
            fhist+=fast_hist((fseg>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),
                      2)
            ahist+=fast_hist((aseg_predict>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),
                      2)
            bhist+=fast_hist((bseg_predict>0.5).cpu().detach().numpy().flatten(), bmask.cpu().numpy().flatten(),
                      2)
            loss_sigma += float(loss.mean().item())

    iou=(np.diag(fhist) / (fhist.sum(axis=1) + fhist.sum(axis=0) - np.diag(fhist)))[-1]
    valloss=loss_sigma / len(val_loader)
    val_best_loss=valloss
    print('Best Loss:'+ str(val_best_loss))
    print('Before IOU:',(np.diag(bhist) / (bhist.sum(axis=1) + bhist.sum(axis=0) - np.diag(bhist)))[-1])
    print('After IOU:',(np.diag(ahist) / (ahist.sum(axis=1) + ahist.sum(axis=0) - np.diag(ahist)))[-1])
    print('Final IOU:',(np.diag(fhist) / (fhist.sum(axis=1) + fhist.sum(axis=0) - np.diag(fhist)))[-1])
no_optim=0
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
    # net.module.beforeUnet.eval()
    # net.module.beforeUnet.requires_grad_(False)
    print('Epoch:[{:0>3}/{:0>3}] '.format(epoch + 1, cfg['epoch'])+'Learning rate '+str(optimizer.param_groups[0]['lr']))
    data_loader=train_loader
    for iter_num, data in enumerate(data_loader):
        (binputs, bmask, bedge), (ainputs, amask, aedge),img_names= data[0]
        trainval=data[1]
        batch_pixel_size = binputs.size(0) * binputs.size(2) * binputs.size(3)
        binputs, bmask, bedge = binputs.cuda(), bmask.cuda(), bedge.cuda()
        ainputs, amask, aedge = ainputs.cuda(), amask.cuda(), aedge.cuda()
        optimizer.zero_grad()
        inputs = [binputs, ainputs]
        gts = [[bmask, bedge], [amask, aedge]]
        loss, outputs = net(inputs, gts,trainval)
        [bseg_predict, bedge_predict], [aseg_predict, aedge_predict], chg, fseg = outputs
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
        print('Validating')
    else:
        continue
    net.eval()
    loss_sigma = 0.0
    cls_num = 2
    hist=0
    fhist=0
    ahist=0
    bhist=0
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
            loss, outputs = net(inputs, gts,True)
            [bseg_predict, bedge_predict], [aseg_predict, aedge_predict],chg, fseg = outputs
            fhist+=fast_hist((fseg>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),
                      2)
            ahist+=fast_hist((aseg_predict>0.5).cpu().detach().numpy().flatten(), amask.cpu().numpy().flatten(),
                      2)
            bhist+=fast_hist((bseg_predict>0.5).cpu().detach().numpy().flatten(), bmask.cpu().numpy().flatten(),
                      2)
            loss_sigma += float(loss.mean().item())
    iou = np.diag(fhist) / (fhist.sum(axis=1) + fhist.sum(axis=0) - np.diag(fhist))
    print("Validating: Epoch:[{:0>3}/{:0>3}] Time:{:0>2}:{:0>2} Validation Loss:{:.4f} Validation {}:{:.4f}".format\
                    (
                epoch + 1, cfg['epoch'],int((datetime.now()-eph_dtime).seconds//60),int((datetime.now()-eph_dtime).seconds%60),
            loss_sigma / len(val_loader),\
            'IOU:',iou[-1]))
    print('Before IOU:',(np.diag(bhist) / (bhist.sum(axis=1) + bhist.sum(axis=0) - np.diag(bhist)))[-1])
    print('After IOU:',(np.diag(ahist) / (ahist.sum(axis=1) + ahist.sum(axis=0) - np.diag(ahist)))[-1])
    print('Final IOU:',(np.diag(fhist) / (fhist.sum(axis=1) + fhist.sum(axis=0) - np.diag(fhist)))[-1])
    print('valloss:',valloss,'---->',loss_sigma / len(val_loader),' -best loss:',val_best_loss)
    valloss = loss_sigma / len(val_loader)
    if valloss >= val_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        val_best_loss = valloss
        if epoch!=0:
            print('Saving model!')
            torch.save(net.module.state_dict(),  os.path.join(cfg.project_path,'models',cfg.save_name))
    if no_optim > 50:
        print('early stop at %d epoch' % epoch)
        break
    if no_optim >=3:
        if float(scheduler.get_lr()[-1]) < 1e-7:
            break
        print('Loading model')
        if os.path.exists( os.path.join(cfg.project_path,'models',cfg.save_name)):
            net.module.load_state_dict(torch.load( os.path.join(cfg.project_path,'models',cfg.save_name)))
        else:
            print('No saved Model! Loading Init Model!')
            net.module.load_state_dict(torch.load( os.path.join(cfg.project_path,'models',cfg.read_name)))
        scheduler.step()
        no_optim=0
print('Finished Training')