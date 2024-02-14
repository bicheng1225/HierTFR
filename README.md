# HierTFR
This repository is the implementation of the paper, An Effective Pronunciation Assessment Approach Leveraging Hierarchical Transformers and Pre-training Strategies (Submitted to ACL2024).
> The development of our code is based on the open-source project at  [https://github.com/YuanGongND/gopt](https://github.com/YuanGongND/gopt) (Gong et al, 2022).

## Dataset
An open source dataset, SpeechOcean762 (licenced with CC BY 4.0) is used. Please refer to the project at [[https://www.openslr.org/101](https://github.com/jimbozhang/speechocean762)https://github.com/jimbozhang/speechocean762].

## Package Requirements
Install the below packages in your virtual environment before running the code.
- python version 3.7
- pytorch version '1.13.1+cu117'
- numpy version 1.20.3
- pandas version 1.5.0

## Pretraining Stage
- `cd src`
- `bash run_preTrain.sh`

## Training and Evaluation
This bash script will load the pre-trained model and train the model 5 times with epoch ([0, 1, 2, 3, 4]).
- `cd src`
- `bash run.sh`

Note that every run does not produce the same results due to the random elements.
