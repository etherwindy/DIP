# DIP

This repo is for the first task of [CGI-HRDC 2023 - Hypertensive Retinopathy Diagnosis Challenge](https://codalab.lisn.upsaclay.fr/competitions/11877#learn_the_details-terms_and_conditions)

## Introduction

Hypertensive retinopathy (HR) refers to retinal damage caused by high blood pressure. Elevated blood pressure initially causes changes in the retina, causing spasmodic constriction of the retinal arteries. If the blood pressure is controlled in time during this period, the damage to the blood vessels is reversible. However, the analysis of hypertensive retinopathy is limited by the manual inspection process of image by image, which is time-consuming and relies heavily on the experience of ophthalmologists. Therefore, an effective computer-aided system is essential to help ophthalmologists analyze the progression of disease.

In order to promote the application of machine learning and deep learning algorithms in computer-aided automatic clinical hypertensive retinopathy diagnosis, we organize the hypertensive retinopathy diagnosis challenge. With this dataset, various algorithms can test their performance and make a fair comparison with other algorithms.

Task 1 is hypertension classification. Given a fundus image of a patient's eye, the task is to confirm whether this patient suffers from hypertension. Category 0 represents no hypertension and category 1 represents hypertension. This is a two-class classification task.

The backbone of our model is ResNet34. Since the given dataset is quite small, we utilized several augmentation methods and introduced SimCLR, a contrastive learning method to pretrain our model. The testing result of our implementation is shown below:

| Kappa | F1 | Specificity | Average | CPU Time |
| --- | --- | --- | --- | --- |
| 0.3472 (6) | 0.6270 (8) | 0.7986 (2) | 0.5909 (5) | 0.1071 (8) |

## Usage

Start by clone the repo:

```bash
git clone https://github.com/etherwindy/DIP
cd DIP
```
Create conda environment:

```bash
conda env create -f environment.yaml
conda activate dip
```

To train a new model, you need to download dataset. We use [SMDG, A Standardized Fundus Glaucoma Dataset | Kaggle](https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset) to pretrain the model:

```bash
kaggle datasets download -d deathtrooper/multichannel-glaucoma-benchmark-dataset
unzip -d SMPG/ multichannel-glaucoma-benchmark-dataset.zip
mv SMPG/full-fundus dataset/pretrain
rm -rf SMPG/ multichannel-glaucoma-benchmark-dataset.zip
```

Then pretrain the model:

```bash
// resnet
python pretrain_resnet.py --batch_size=64 --epochs=100 --temperature=0.07 --lr=1e-4 --min_lr=1e-5 --warmup_epochs=10 --weight_decay=1e-5 --img_siz=(512,512) --gpu=0
// vit
python pretrain_vit.py --model=mae_vit_base_patch16 --mask_ratio=0.75 --accum_iter=1 --warmup_epochs=10 --batch_size=128 --epochs=100 --lr=1e-4 --min_lr=1e-5 --weight_decay=1e-5 --gpu=0
// vgg
python pretrain_vgg.py --batch_size=64 --epochs=100 --temperature=0.07 --lr=1e-4 --min_lr=1e-5 --warmup_epochs=10 --weight_decay=1e-5 --img_size=(224,224) --gpu=0
// densenet
python pretrain_densenet.py --batch_size=64 --epochs=20 --temperature=0.07 --lr=1e-4 --min_lr=1e-5 --warmup_epochs=4 --weight_decay=1e-5 --img_siz=(224,224) --gpu=0
// efficientnet
python pretrain_efficientnet.py --batch_size=64 --epochs=20 --temperature=0.07 --lr=1e-4 --min_lr=1e-5 --warmup_epochs=4 --weight_decay=1e-5 --img_siz=(224,224) --gpu=0
```

In fine-tuning stage, first you need to download the competition labeled training set. Put the images under `dataset/image_original` file (The path of images are `dataset/image_original/*.png`). Then rename the CSV file  `label_original.csv` and put it under `dataset/`. Then split and augment the dataset:

```bash
cd dataset
python split_data.py
python augmentation_flip.py
python augmentation_light.py
cd ../
```

Then finetune the pretrained model:

```bash
// resnet
python finetune_resnet.py --img_size=(512,512) --batch_size=64 --val_batch_size=64 --epochs=100 --warmup_epochs=4 --lr=1e-4 --min_lr=1e-7 --weight_decay=1e-5 --gpu=0
// vit
python finetune_vit.py --model=vit_base_patch16 --bce --nb_classes=1 --global_pool='avg' --drop_path=0.1 --mask_ratio=0.75  --accum_iter=1 --epochs=100 --warmup_epochs=2 --batch_size=64 --val_batch_size=64 --lr=1e-4 --min_lr=1d-7 --weight_decay=1e-5 --gpu=0
// vgg
python finetune_vgg.py --img_size=(224,224) --batch_size=64 --val_batch_size=64 --epochs=100 --warmup_epochs=4 --lr=1e-4 --min_lr=1e-7 --weight_decay=1e-5 --gpu=0
// densenet
python finetune_densenet.py --img_size=(224,224) --batch_size=64 --val_batch_size=64 --epochs=100 --warmup_epochs=4 --lr=1e-4 --min_lr=1e-7 --weight_decay=1e-5 --gpu=0
// efficientnet
python finetune_efficientnet.py --img_size=(224,224) --batch_size=64 --val_batch_size=64 --epochs=100 --warmup_epochs=4 --lr=1e-4 --min_lr=1e-7 --weight_decay=1e-5 --gpu=0
```

Model weights of best model and latest model be stored in `output` folder. To submit your model, you need to move your model file to `submit/` (for VIT model, move it to submit_vit/) and rename it as 'model_weights.pth'. Then zip all files in the folder. `submit/model.py` is defaultly set for ResNet. If you want to submit other model, remember to update your code.
