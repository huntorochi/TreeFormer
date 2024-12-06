# TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation (WACV2025)

## Xinpeng Liu<sup>1</sup>, Hiroaki Santo<sup>1</sup>, Yosuke Toda<sup>2,3</sup>, Fumio Okura<sup>1</sup><br> (<sup>1</sup> Osaka University, <sup>2</sup> Phytometrics, <sup>3</sup> Nagoya University)
[![arXiv](https://img.shields.io/badge/arXiv-2411.16132-b31b1b.svg)](https://arxiv.org/abs/2411.16132) 

## Requirements
* CUDA>=9.2
* PyTorch>=1.7.1

For other system requirements please follow

```bash
pip install -r requirements.txt
```

### Compiling CUDA operators
```bash
cd ./models/ops
python setup.py bulid install
```


## Code Usage

## 1. Dataset preparation

Please download [Guyot dataset] by following the steps under **Usage**. The structure of the dataset should be as follows:

```
guyot_data/
└── train/
    └── check/
        └── images
    └── data/
        └── Set02_IMG_3468.pt
    └── img/
        └── images
    └── unet/
        └── images
└── test/
    └── check/
        └── images
    └── data/
        └── Set02_IMG_3468.pt
    └── img/
        └── images
    └── unet/
        └── images
└── val/
    └── check/
        └── images
    └── data/
        └── Set02_IMG_3468.pt
    └── img/
        └── images
    └── unet/
        └── images
```

## 2. Training

#### 2.1 Prepare config file

The config file can be found at `.configs/tree_2D_use_mst_only1.yaml` and `.configs/tree_2D_use_unmst_only1.yaml`. Make custom changes if necessary.

#### 2.2 Train

For example, the command for training Relationformer is following:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/tree_2D_use_mst_only1.yaml --cuda_visible_device 0 1 2 3 4 5 6 7 
```
```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/tree_2D_use_mst_only1.yaml --cuda_visible_device 0 1 2 3 4 5 6 7 --resume trained_weights/check/checkpoint_81_epoch.pkl 
```

## 3. Evaluation

Once you have the config file and trained model, run following command to evaluate it on test set:

```bash
python valid_smd_guyot_nx.py
```

## 4. Citation

```
@inproceedings{liu2025treeformer,
  title={{TreeFormer}: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation},
  author={Liu, Xinpeng and Santo, Hiroaki and Toda, Yosuke and Okura, Fumio},
  booktitle={Proceedings of IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```
