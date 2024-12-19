<h1 align="center"> Arbitrary Reading Order Scene Text Spotter with Local Semantics Guidance </h1>

<p align="center">
<a href="https://arxiv.org/pdf/2412.10159"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

<p align="center">
  <a href="## news">News</a> |
  <a href="## Installation">Installation</a> |
  <a href="## Datasets">Datasets</a> |
  <a href="## Training">Training</a> |
  <a href="## Inference">Inference</a> |
  <a href="## Acknowledgement">Acknowledgement</a>
</p>

This is the official repo for the paper "Arbitrary Reading Order Scene Text Spotter with Local Semantics Guidance", which is accepted to AAAI 2025.

We will release our code soon.


## News

***2024/12/19***
- Update the README!

***2024/12/15***
- Update the arxiv version!

***2024/12/10***
- The paper is accepted by AAAI 2025! 

## Installation
```
conda create -n lsg python=3.9 -y
conda activate lsg
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv-full==1.5.2
mim install mmdet==2.28.2
git clone https://github.com/pd162/LSG
cd LSG
pip install -e .
```
## Datasets
The convert annotations can be download from [Google Drive], Please download and extract the above datasets into the `data` folder following the file structure below.

```
data
├─totaltext
│  │ totaltext_train.json
│  │ totaltext_test.json
│  └─imgs
│      ├─training
│      └─test
├─CTW1500
│  │ instances_training.json
│  │ instance_test.json
│  └─imgs
│      ├─training
│      └─test
├─mlt
│  │  train_polygon.json
│  └─images
├─synthtext-150k
      ├─syntext1
      │  │  train_polygon.json
      │  └─images
      ├─syntext2
         │  train_polygon.json
         └─images
```

## Training
`CUDA_VISIBLE_DEVICES=0,1 ./tools/train.sh config/LSG/lsg_pretrain.py work_dirs/pretrain 2
`
## Inference
`CUDA_VISIBLE_DEVICES=0 python tools/test.py config/tpsnet/tpsnet_totaltext.py work_dirs/totaltext/latest.pth --eval eval_text`

## Acknowledgement
We sincerely thank [MMOCR](https://github.com/open-mmlab/mmocr), [TPSNet](https://github.com/Wei-ucas/TPSNet) for their excellent works.
