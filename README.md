# HOICLIP-reproduce

The final project for Vision and Language 2023 @ PKU.

Reproduction of the CVPR 2023 paper:"[HOICLIP: Efficient-Knowledge-Transfer-for-HOI-Detection-with-Visual-Linguistic-Model](https://arxiv.org/abs/2303.15786)".

Reproduced by [Wei Xia](https://github.com/adventurexw) and [Yifan Huang](https://github.com/AllenYolk).
Contributed by Shan Ning*, Longtian Qiu*, Yongfei Liu, Xuming He.

For more details of the original paper, please refer to the [original repository](https://github.com/Artanic30/HOICLIP).

![model](paper_images/intro.png)

## Installation

Install the dependencies.

```shell
pip install -r requirements.txt
```

## Data preparation

### HICO-DET

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After
finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The
annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The
downloaded annotation files have to be placed as follows.
For fractional data setting, we provide the
annotations [here](https://drive.google.com/file/d/13O_uUv_17-Db9ghDqo4z2s3MZlfZJtgi/view?usp=sharing). After
decompress, the files should be placed under `data/hico_20160224_det/annotations`.

```plaintext
data
 └─ hico_20160224_det
     |─ annotations
     |   |─ trainval_hico.json
     |   |─ test_hico.json
     |   |─ corre_hico.json
     |   |─ trainval_hico_5%.json
     |   |─ trainval_hico_15%.json
     |   |─ trainval_hico_25%.json
     |   └─ trainval_hico_50%.json
     :
```

### V-COCO

First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to
generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle`
from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make
directories as follows.

```plaintext
GEN-VLKT
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```

For our implementation, the annotation file have to be converted to the HOIA format. The conversion can be conducted as
follows.

```shell
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```

Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error
with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be
generated to `annotations` directory.

## Pre-trained model

Download the pretrained model of DETR detector for [ResNet50](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth)
, and put it to the `params` directory.

```shell
python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-hico.pth \
        --num_queries 64

python ./tools/convert_parameters.py \
        --load_path params/detr-r50-e632da11.pth \
        --save_path params/detr-r50-pre-2branch-vcoco.pth \
        --dataset vcoco \
        --num_queries 64
```

## Training

After the preparation, you can start training with the following commands.

Here, we only list the commands for our reproduction and improvements.
Please refer to the [original repository](https://github.com/Artanic30/HOICLIP) for a complete list of all the experiments.

### Generate our verb representations for HICO-Det

```shell
sh ./scripts/generate_verb.sh
```

We provide the generated verb representation:

* `./tmp/verb.pth` is the original verb representation for hico-det.
* `./tmp/verb_hico_ours.pth` is our verb representation for hico-det.
* `./tmp/vcoco_verb.pth` is for vcoco. It is provided by the authors of the original work, and cannot be generated manually.

### Training V-COCO

```shell
sh ./scripts/train_vcoco.sh
```

### Training 5% HICO-DET

```shell
# default setting
sh ./scripts/train_hico_frac_baseline.sh
# with our improvement on verb representation
sh ./scripts/train_hico_frac_our_verb.sh
# with our improvement on Q_{inter}
sh ./scripts/train_hico_frac_our_mdoel.sh
```

## Evaluation

### Evaluating V-COCO

```shell
# default setting
sh ./scripts/eval_vcoco.sh
```

### Evaluating HICO-DET

```shell
# default setting
sh ./scripts/eval_hico_frac_baseline.sh
# with our improvement on verb representation
sh ./scripts/eval_hico_frac_our_verb.sh
# with our improvement on Q_{inter}
sh ./scripts/eval_hico_frac_our_mdoel.sh
```

### Training Free Enhancement

The `Training Free Enhancement` is used when args.training_free_enhancement_path is not empty.
The results are placed in args.output_dir/args.training_free_enhancement_path.
By default, we set the topk to `[10]` for HICO-Det, and `[10, 20]` for V-COCO.

## Regular HOI Detection Results

Here are the results in the original paper.

For results of our reproduction, please read `report.pdf`.

### HICO-DET

|         | Full (D) |Rare (D)|Non-rare (D)|Full(KO)|Rare (KO)|Non-rare (KO)|Download|              Conifg               |
|:--------|:--------:| :---: | :---: | :---: |:-------:|:-----------:| :---: |:---------------------------------:|
| HOICLIP |  34.69   | 31.12 |35.74 | 37.61|  34.47  |    38.54    | [model](https://drive.google.com/file/d/1q3JuEzICoppij3Wce9QfwZ1k9a4HZ9or/view?usp=drive_link) | [config](./scripts/train_hico.sh) |

D: Default, KO: Known object. The best result is achieved with training free enhancement (topk=10).

### HICO-DET Fractional Setting

| | Fractional |Full| Rare | Non-rare  |                     Config                     | 
| :--- |:----------:| :---: |:----:|:---------:|:----------------------------------------------:|
| HOICLIP|     5%     |22.64 |21.94 |   24.28   |     [config](./scripts/train_hico_frac.sh)     |
| HOICLIP|    15%     |27.07 |   24.59   |   29.38   |     [config](./scripts/train_hico_frac.sh)     |
| HOICLIP|    25%     |28.44 |25.47|   30.52   |     [config](./scripts/train_hico_frac.sh)     |
| HOICLIP|    50%     |30.88|26.05 |   32.97   |     [config](./scripts/train_hico_frac.sh)     |

You may need to change the `--frac [portion]%` in the scripts.

### V-COCO

| | Scenario 1 | Scenario 2 | Download |               Config               | 
| :--- | :---: | :---: | :---: |:----------------------------------:|
|HOICLIP| 63.50| 64.81 | [model](https://drive.google.com/file/d/1PAT2P3TaBCwG3AHuFcbe3iOk2__XOf_R/view?usp=drive_link) | [config](./scripts/train_vcoco.sh) |

## Zero-shot HOI Detection Results

| |Type |Unseen| Seen| Full|Download|                   Conifg                   |
| :--- | :---: | :---: | :---: | :---: | :---: |:------------------------------------------:|
| HOICLIP|RF-UC |25.53 |34.85 |32.99| [model](https://drive.google.com/file/d/1E7QLhKgsC1qutGUinXIPmANRR3glYQ1h/view?usp=sharing)|  [config](./scripts/train_hico_rf_uc.sh)   |
| HOICLIP|NF-UC |26.39| 28.10| 27.75| [model](https://drive.google.com/file/d/1W1zUEX3uDJN32UMI8seTDzmXZ5i9uBz7/view?usp=drive_link)|  [config](./scripts/train_hico_nrf_uc.sh)  |
| HOICLIP|UO |16.20| 30.99| 28.53| [model](https://drive.google.com/file/d/1oOe8rOwGDugIhd5N3-dlwyf5SpxYFkHE/view?usp=drive_link)|    [config](./scripts/train_hico_uo.sh)    |
| HOICLIP|UV|24.30| 32.19| 31.09| [model](https://drive.google.com/file/d/174J4x0LovEZBnZ_0yAObMsdl5sW9SZ84/view?usp=drive_link)|    [config](./scripts/train_hico_uv.sh)    |

## Citation

Please consider citing our paper if it helps your research.

```
@inproceedings{ning2023hoiclip,
  title={HOICLIP: Efficient Knowledge Transfer for HOI Detection with Vision-Language Models},
  author={Ning, Shan and Qiu, Longtian and Liu, Yongfei and He, Xuming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23507--23517},
  year={2023}
}
```

## Acknowledge

Codes are built from [HOICLIP](https://github.com/Artanic30/HOICLIP), [GEN-VLKT](https://github.com/YueLiao/gen-vlkt), [PPDM](https://github.com/YueLiao/PPDM)
, [DETR](https://github.com/facebookresearch/detr), [QPIC](https://github.com/hitachi-rd-cv/qpic)
and [CDN](https://github.com/YueLiao/CDN). We thank them for their contributions.
