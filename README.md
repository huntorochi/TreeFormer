# TreeFormer
TreeFormer: Single-view Plant Skeleton Estimation via Tree-constrained Graph Generation

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
python test.py --config configs/road_2D.yaml --cuda_visible_device 3 --checkpoint ./trained_weights/last_checkpoint.pt
```
