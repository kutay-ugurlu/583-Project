# Analysis of Deep MLP SER 
## The unmodified code represented in [Deep Multilayer Perceptrons for Dimensional Speech Emotion Recognition](https://github.com/bagustris/deep_mlp_ser)
### This is the project repository of EE583 project where the study cited above is evaluated with other corpora.

## Initial Setup
#### The data is transferred to the Google Drive due to Git LFS restrictions.
#### One should download the folders in the One Drive URL below and __REPLACE__ with the corresponding folders in the GitHub repository before running the scripts to not get errors. 
[MELD Dataset](https://1028f8d26f624cd18d39-my.sharepoint.com/:f:/g/personal/kutay_ugurlu_metu_edu_tr/EkSnJAtA9fBCntQzBIEHvCoBhmel9vTzjVsLdz8I6v1Vcg?e=Mgdq43)
##### The conda environment deepmlpser should be created by using the command:
```
conda create --name deepmlpser --file requirements.txt
```

## Running Scripts 
### MP4 to WAV
* After MELD Raw data is replaced with the one provided in the link above, one can find MP4 and WAV files. One may also reproduce the WAV files by running [MP4_to_WAV.py](https://github.com/kutay-ugurlu/Analysis-of-Deep-MLP-SER/blob/master/data/MELDRaw/test_data/output_repeated_splits_test/mp4_to_wav.py) which requires [FFMPEG](https://www.ffmpeg.org/).
### Feature generation
* After WAV files are created, run [feat_extract.py](https://github.com/kutay-ugurlu/Analysis-of-Deep-MLP-SER/blob/master/data/MELDRaw/test_data/output_repeated_splits_test/WAVs/feat_extract.py) to get the features in numpy files.
### Label generation
* Run [read_csv.ipynb](https://github.com/kutay-ugurlu/Analysis-of-Deep-MLP-SER/blob/master/data/MELDRaw/read_csv.ipynb) to generate labels.
### Model training and testing 
* Run [run_all.py](https://github.com/kutay-ugurlu/Analysis-of-Deep-MLP-SER/blob/master/code/run_all.py) script to reproduce all results. The results will be saved in the JSONs folder.
