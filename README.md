
# Automatic Body Part Identification in Whole-Body MRI

## What does it do?
**Body part identification**: Find body parts in 3D MRI images. Body parts of interest are the wrists, shoulder joints, diaphragm/liver dome, hips, ankles.

**Matlab labeling app**: Helps to manually label the body parts in 3D MRI images in coronal view.

## About
This code is part of my Bachelor Thesis, conducted at the [Institute of Signal Processing and System Theory](https://www.iss.uni-stuttgart.de/en/) of the University of Stuttgart in collaboration with the [Medical Image and Data Analysis Lab](http://www.midaslab.org/) of the University of Tübingen in 2018/2019. It is a part of a larger project for semantic segmentation of medical images, which can be found [here](https://github.com/lab-midas/med_segmentation). Due to patient and study data protection, **I cannot realease any data or visualizations.** Only recently I received permission to publish this code.

## Abstract
The analysis of medical imaging data such as Magnetic Resonance Imaging (MRI) of large cohort studies is a very time consuming process. Therefore automated data analysis software is developed to support physicians and relief their workload. This work demonstrates a Deep Learning (DL) approach to find different sections of the human body in MRI images that are needed to evaluate Adipose Tissue (AT) distributions for diabetes mellitus type 2. The architecture of the Neural Network (NN) is based on Zhang<sup>[1](#Zhang)</sup> and uses patches. The network is adapted to fit the task and is designed with a hybrid output that combines classification and regression and is inspired by HCR-Net<sup>[2](#Wang)</sup>. The model is trained and its hyperparameters are optimized using a grid search. The trained and optimized model is able to precisely predict separating lines between body parts on the test data.

<a name="Zhang">1</a>: J. Zhang, M. Liu and D. Shen, “Detecting anatomical landmarks from limited medical imaging data using two-stage task-oriented deep neural networks,” _IEEE Transactions on Image Processing,_ vol. PP, pp. 1–1, 06 2017.

<a name="Wang">2</a>: Z. Wang, W. Li, Y. Kao, D. Zou, Q. Wang, M. Ahn and S. Hong, “Hcr-net: A hybrid of classification and regression network for object pose estimation,” in _Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18. International Joint Conferences on Artificial Intelligence Organization,_ 7 2018, pp. 1014–1020. [Online]. Available: https://doi.org/10.24963/ijcai.2018/141

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
