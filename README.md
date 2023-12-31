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

If you only want to use our trained model, just run:

```bash
python test.py -i <image dir path> -o <output csv file>
```

For example:

```bash
python test.py -i ./dataset/image/ -o ./test.csv
```

The program will process all PNG images in the input folder and write the results in a certain CSV file.

To train a new model, you need to download dataset. We use [SMDG, A Standardized Fundus Glaucoma Dataset | Kaggle](https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset) to pretrain the model:

```bash
kaggle datasets download -d deathtrooper/multichannel-glaucoma-benchmark-dataset
unzip -d SMPG/ multichannel-glaucoma-benchmark-dataset.zip
mv SMPG/full-fundus dataset/pretrain
rm -rf SMPG/ multichannel-glaucoma-benchmark-dataset.zip
```

Then pretrain the model:

```bash
python pretrain.py
```

In fine-tuning stage, first you need to download the competition labeled training set. Put the images under `dataset/image` file (`dataset/image/*.png`) and the CSV file should be `dataset/label.csv`. Then augment the dataset:

```bash
cd dataset
python augmentation_flip.py
python augmentation_light.py
cd ../
```

Then fine tune the pretrained model:

```bash
python main.py
```

Model weights of each epoch will all be stored in `output` folder. You can choose the one you like:

```bash
cp output/epoch_<select an epoch>.pth submit/model_weights.pth
```

For example:

```bash
cp output/epoch_19.pth submit/model_weights.pth
```

Then you can use your new model:

```bash
python test.py -i <image dir path> -o <output csv file>
```
