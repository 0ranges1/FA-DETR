# FA-DETR: Foreground Attentive DEtection TRansformer for Few-Shot Object Detection

-------


## Installation

### Pre-Requisites
Environment setups:
- Ubuntu LTS 18.04
- 8x NVIDIA V100 GPUs (32GB)
- CUDA 10.2
- Python == 3.7
- PyTorch == 1.7.1+cu102, TorchVision == 0.8.2+cu102
- GCC == 7.5.0
- cython, pycocotools, tqdm, scipy

&nbsp;

### Compile [Deformable Attention](https://github.com/fundamentalvision/Deformable-DETR)

As FA-DETR is developed upon Deformable DETR, you need to compile Deformable Attention first.
```bash
# compile CUDA operators of Deformable Attention
cd FA-DETR
cd ./models/ops
sh ./make.sh
python test.py  # unit test (should see all checking is True)
```

&nbsp;

### Data Preparation

Please download [MSCOCO 2017 dataset](https://cocodataset.org/) and [Pascal VOC dataset](https://drive.google.com/file/d/1JCxJ2lmNX5E4YsvAZnngVZ5hQeJU67tj/view?usp=sharing) first and organize them as following:


```
FA-DETR/
└── data/
    ├── coco/                   # MS-COCO dataset
    │   ├── train2017/
    │   ├── val2017/
    │   └── annotations/
    │       ├── instances_train2017.json
    │       └── instances_val2017.json
    ├── coco_fewshot/           # Few-shot dataset 
    │
    ├── voc/                    # MS-COCO-Style Pascal 
    │   ├── images/
    │   └── annotations/
    │       ├── xxxxx.json
    │       ├── yyyyy.json
    │       └── zzzzz.json
    ├── voc_fewshot_split1/     # VOC Few-shot dataset
    ├── voc_fewshot_split2/     # VOC Few-shot dataset
    └── voc_fewshot_split3/     # VOC Few-shot dataset
    
```

The [`coco_fewshot`](data/coco_fewshot), [`voc_fewshot_split1`](data/voc_fewshot_split1), [`voc_fewshot_split2`](data/voc_fewshot_split2) and [`voc_fewshot_split3`](data/voc_fewshot_split3) folders contain randomly sampled few-shot datasets under different seeds and K-shot setup. We ensure that there are exactly K object instances for each novel category. The numbers of base-class object instances vary. 

Note that we have transformed the original Pascal VOC dataset format into MS-COCO format for parsing.

&nbsp;

## Usage
Scripts are stored in [`./scripts`](scripts). The arguments are pretty easy and straightforward to understand. 
### To Perform _**Base Training**_

Taking MS-COCO as an example, run the following commands to perform base training:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/base_training_coco.sh
```

For Pascal Voc, run the following commands:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/base_training_voc.sh
```

### To Perform _**Few-Shot Finetuning**_
For MS-COCO, run the following commands:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/fewshot_finetuning_coco.sh
```

For Pascal VOC, run the following commands:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/fewshot_finetuning_voc.sh
```

Note that you need to specify the number of shots, few-shot random seed, training epoch setups, and the checkpoint file path in the [`fewshot_finetuning_coco.sh`](scripts/fewshot_finetuning_coco.sh) and [`fewshot_finetuning_voc.sh`](scripts/fewshot_finetuning_voc.sh). In addition, you also need to specify the partition of Pascal VOC in [`fewshot_finetuning_voc.sh`](scripts/fewshot_finetuning_voc.sh) (i.e., voc1, voc2 or voc3).

### To Perform _**Only Inference**_ After Few-Shot Finetuning

We take Pascal VOC as an example. Simply run the following commands:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/inference_ft_voc.sh
```

For MS-COCO, it follows the same approach. Users can easily generate the `inference_ft_coco.sh` based on the [`inference_ft_voc.sh`](scripts/inference_ft_voc.sh) and [`fewshot_finetuning_coco.sh`](scripts/fewshot_finetuning_coco.sh). Then, simply run:
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./scripts/inference_ft_coco.sh
```

Note that user should set `--eval` and `--resume path/to/checkpoint.pth/generated/by/few-shot-fintuning` correctly.