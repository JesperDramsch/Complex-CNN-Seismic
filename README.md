# Complex-CNN-Seismic
This repository reproduces "Complex-valued neural networks for machine learning on non-stationary physical data".

## Data
Obtained from https://github.com/olivesgatech/facies_classification_benchmark via
```
# download the files: 
wget https://zenodo.org/record/3755060/files/data.zip
# check that the md5 checksum matches: 
openssl dgst -md5 data.zip # Make sure the result looks like this: MD5(data.zip)= bc5932279831a95c0b244fd765376d85, otherwise the downloaded data.zip is corrupted. 
```

Preparation for training via `src/data_prep.py`.

## Training
Training done on GPU cluster using `src/mass_train.py`.

## Prediction
Use trained models to generate predictions `src/save_predictions.py`.

## Analysis
Numerical and qualitative analysis generated via `src/explore.py`.
