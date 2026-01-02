# Comparative Assessment of CNN and Transformer U-Nets in Multiple Sclerosis Lesion Segmentation

This repository contains the implementation of our paper [__Comparative Assessment of CNN and Transformer U-Nets in Multiple Sclerosis Lesion Segmentation__](https://onlinelibrary.wiley.com/doi/abs/10.1002/ima.70146).

## Abstract

Multiple sclerosis (MS) is a chronic autoimmune disease that causes lesions in the central nervous system. Accurate segmentation and quantification of these lesions are essential for monitoring disease progression and evaluating treatments. Several architectures are used for such studies, the most popular being U-Net-based models. Therefore, this study compares CNN-based and Transformer-based U-Net architectures for MS lesion segmentation. Six U-Net architectures based on CNN and transformer, namely U-Net, R2U-Net, V-Net, Attention U-Net, TransUNet, and SwinUNet, were trained and evaluated on two MS datasets, ISBI2015 and MSSEG2016. T1-w, T2-w, and FLAIR sequences were jointly used to obtain more detailed features. A hybrid loss function, which involves the addition of focal Tversky and Dice losses, was exploited to improve the performance of models. This study was carried out in three steps. First, each model was trained separately and evaluated on each dataset. Second, each model was trained on the ISBI2015 dataset and evaluated on the MSSEG2016 dataset and vice versa. Finally, these two datasets were combined to increase the training samples and assessed on the ISBI2015 dataset. Accordingly, the R2U-Net and the V-Net models (CNN-based) achieved the highest ISBI scores among the other models. The R2U-Net model achieved the highest ISBI scores in the first and last steps with average scores of 92.82 and 92.91, while the V-Net model achieved the highest ISBI score in the second step with an average score of 91.28. Our results show that CNN-based models surpass the Transformer-based U-Net models in most metrics for MS lesion segmentation.

## Generated NPY files

After generating NPY files from the source of ISBI2015 and MSSEG2016 training datasets, these files should be placed into the dataset folder as follows:

```
dataset
|───isbi2015
|   | rater_1_train_images.npy
|   | rater_1_train_masks.npy
|   | rater_2_train_images.npy
|   | rater_2_train_masks.npy
|───msseg2016
|   | train_images.npy
|   | train_masks.npy
```

## How to train

To get all options, use -h or --help

```
python train.py -h
```

Here is the training example for the ISBI2015 dataset. Model name can be "unet_2d", "vnet_2d", "r2_unet_2d", "att_unet_2d", "transunet_2d", "swin_unet_2d". Default is "unet_2d"

```
python train.py --dataset_type isbi2015 --model_type unet_2d --validation_split_percentage 0.10 --batch_size 8 --epochs 300
```

## How to cite:

If you use this repository, please cite this study as given:

```
 @article{sarica2024comparison,
    title={CNN and Transformer U-Nets in Multiple Sclerosis Lesion Segmentation: A Comparative Assessment},
    author={Sarica, Beytullah, Bicakci, Yunus Serhat and Seker, Dursun Zafer},
    journal={},
    volume={},
    pages={},
    year={},
    publisher={}
  } 
```
