from core.nets.SGEPUNet import EPUnet,SGEPUnet
import torch
import torch.nn as nn
import os
net1=EPUnet(1,nn.BCEWithLogitsLoss())
net2=SGEPUnet(1,nn.BCEWithLogitsLoss())
read_modelname='ECUNet17.pth'
save_modelname='MultiSCRUNet_init.pth'
if os.path.exists(os.path.join('../models',read_modelname)):
    print('Loading ',read_modelname)
    pre_validate=True
    ECUpretrained_dict=torch.load(os.path.join('../models', read_modelname))
    net1.load_state_dict(ECUpretrained_dict)
    print('ECUNet Loaded!')
else:
    print('Model for ECUNet not exist!')
    assert NameError

Multidict = {'beforeUnet.'+k: v for k, v in ECUpretrained_dict.items()}
Multidict.update({'afterUnet.'+k: v for k, v in ECUpretrained_dict.items()})
pretrained_dict = {k: v for k, v in Multidict.items() if k in net2.state_dict()}
net2_dict=net2.state_dict()
net2_dict.update(pretrained_dict)
net2.load_state_dict(net2_dict)
torch.save(net2.state_dict(), os.path.join('../models',save_modelname))
print('MultiECUNet Initialize Success! Model have been saved to',save_modelname)