Deep Building Footprint Update Network: A Semi-supervised Method for Updating Existing Building Footprint from bi-temporal remote sensing images
====  
The manuscript
----

Building footprint information is one foundation for understanding urban processes and hence a program for environmentally sustainable urbanization. For most cities, municipal governments have constructed basic building contour databases for basic land resource management and urban planning. These building contour databases, however, need to be updated regularly to ensure their validity. Cities are dynamic, changing rapidly over short periods, thus there is an urgent need for automatic methods to produce up-to-date building footprints. Even fully supervised automated methods have major limitations in the building footprint updating process because the labels for current buildings or changed areas are lacking. Moreover, state-of-the-art automatic schemes even some based on deep learning approaches, however, often fail to precisely depict building boundaries, jeopardizing effective updating of building databases. To address these interrelated problems, we developed a novel algorithm to automatically update the existing building databases to their current status with minimal manual intervention; the saliency-guided edge-preservation framework (named SG-EPUNet) for updating existing building databases and producing up-to-date building footprints. This neural network-based method can update building footprints from remote sensing images, requiring manual annotations for only 20% of the buildings. To robustly extract building features, we designed an edge-preservation neural network (EPUNet) that combines edge detection with contextual aggregation in the proposed SG-EPUNet framework. We applied the methods on two datasets from Christchurch, New Zealand, and Guangzhou, China, to assess the effectiveness of the method in urban and suburban areas. For both study areas, the F1-score values for the building update results reached approximately 97%, which confirmed the applicability of the SG-EPUNet framework. The results also show that SG-EPUNet maintains robustness to large areas, even with very limited ground truth labeling. The joint combination of building prior and very high-resolution remote sensing images makes the proposed method a promising approach for automatically producing up-to-date building footprints in practical applications.

Dataset
----

[WHU Building Dataset](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

The code
----
### Requirements
* torch
* torchvision
* pillow
* cv2
* gdal

### Usage
Clone the repository:git clone https://github.com/HaonanGuo/SG-EPUNet.git
1. Run [s2_Singletraining.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s2_Singletraining.py) to train EPU-Net
2. Run [s3_SGEPUNet_Initializer.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s3_SGEPUNet_Initializer.py) to initial SG-EPUNet with EPUNet
3. Run [s4_Multitraining.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s4_Multitraining.py) to train the initialized SG-EPUNet
4. Run [s5_Multipredicting.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s5_Multipredicting.py) to predict the building update result

Help
----
Any question? Please contact us with: guohnwhu@163.com
