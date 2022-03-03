from core.utils.AttrDict import AttrDict
__C = AttrDict()
cfg = __C
# # # Default Network Configurations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
__C.COLOR_MAP=[0,1]
__C.NAME_MAP=['non-building','building']
__C.data_dir='/data/haonan.guo/Dataset'
__C.read_name=''
__C.save_name='EPUNet.pth'
__C.train_txt='all.txt'
__C.val_txt='test.txt'
__C.target_size=(256,256,3)
__C.crop=False
__C.crop_size=(256,256)
__C.warmup_epochs=5
__C.n_class=2
__C.worker_num=24
__C.batch_size=24
__C.epoch=60
__C.base_lr=1e-3
__C.dataset='WHU_Building'
__C.project_path='/data/haonan.guo/BuildingChange_Pytorch'
__C.logging=True
__C.test_mode=False
__C.pre_validate=True

# # Augmentation configurations
__C .AUGMENTATION_CONFIG=AttrDict()
__C.AUGMENTATION_CONFIG.    rotation_range=360  #default:0
__C.AUGMENTATION_CONFIG.    width_shift_range=0.15,    #default:0
__C.AUGMENTATION_CONFIG.    height_shift_range=0.15,     #default:0
__C.AUGMENTATION_CONFIG.    brightness_range=[1,1],  #default:[1,1]
__C.AUGMENTATION_CONFIG.    zoom_range=[0.85,1.15],        #default:[1,1]
__C.AUGMENTATION_CONFIG.    brightness_shift=False,  #default:False
__C.AUGMENTATION_CONFIG.    channel_shift_range=0,   #default:0
__C.AUGMENTATION_CONFIG.    horizontal_flip=True,   #default:False
__C.AUGMENTATION_CONFIG.    vertical_flip=True,     #default:False
__C.AUGMENTATION_CONFIG.    use_random_stretch=False, #default:False
__C.AUGMENTATION_CONFIG.    use_add_noise=False,      #default:False
__C.AUGMENTATION_CONFIG.    use_blur=False,           #default:False
__C.AUGMENTATION_CONFIG.    use_gamma_transform=False, #default:False
__C.AUGMENTATION_CONFIG.    hsv_strengthen=False

# # # Training Configurations
__C .TRAINING_CONFIG =AttrDict()
__C .TRAINING_CONFIG .WEIGHT_DECAY = 0
__C .TRAINING_CONFIG .KERNEL_INITIALIZER = "kaiming_normal"
__C .TRAINING_CONFIG .BN_EPSILON = 1e-3
__C .TRAINING_CONFIG .BN_MOMENTUM = 0.9

