Deep Building Footprint Update Network: A Semi-supervised Method for Updating Existing Building Footprint from bi-temporal remote sensing images
====  
The manuscript
----

### The manuscript is now underreview. <br> 
Building footprint information is the key factor for understanding urban process regimes to promote sustainable urbanization. For most cities, the governments have constructed their basic building contour database for supporting basic land resource management and sustainable urban planning. These building contour databases need to be updated regularly to ensure their validity: moreover, it is important to develop automatic methods to produce up-to-date building footprint information. Due to the lack of labels of current buildings or changed areas in the process of building footprint update, fully supervised methods would show major limitations. Further, state-of-the-art automatic schemes (eventually based on deep learning approach) often fail to precisely depict building boundary, jeopardizing the effective update of building databases. In this study, we attempted to automatically update the existing building databases to their currency with minimal manual intervention. We propose a novel saliency-guided edge-preservation framework (named SG-EPUNet) for updating the existing building databases and producing up-to-date building footprints. This network can automatically update building footprints from remote sensing images with 80% less manual intervention. To extract robust building features, we designed an edge-preservation network (EPU-Net) that combines edge detection with contextual aggregation in the proposed SG-EPUNet. To assess the effectiveness of the method on various urban and suburban scenes, we applied the methods on two datasets from Christchurch, New Zealand, and Guangzhou, China, which are experiencing rapid urbanization. For both study areas, the F1-score values of the building update results reached approximately 97%, which confirmed the applicability of the building footprint update results. The results show also that SG-EPUNet can maintain outstanding robustness to large areas, even with very limited ground truth labeling. The joint combination of building prior and very high-resolution remote sensing images make the proposed method a promising approach for automatically producing up-to-date building contour information in practical applications. 

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

### Usage
Clone the repository:git clone https://github.com/HaonanGuo/SG-EPUNet.git
1. Run [s1_dataset_generator.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s1_dataset_generator.py) to generate dataset
2. Run [s2_Singletraining.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s2_Singletraining.py) to train EPU-Net
3. Run [s3_SGEPUNet_Initializer.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s3_SGEPUNet_Initializer.py) to initial SG-EPUNet with EPUNet
4. Run [s4_Multitraining.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s4_Multitraining.py) to train the initialized SG-EPUNet
5. Run [s5_Multipredicting.py](https://github.com/HaonanGuo/SG-EPUNet/blob/main/s5_Multipredicting.py) to predict the building update result

Help
----
Any question? Please contact us with: guohnwhu@163.com
