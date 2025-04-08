# Attacking Attention of Foundation Models Effectively Disrupts Downstream Tasks

Official PyTorch implementation of the paper ***"Attacking Attention of Foundation Models Effectively Disrupts Downstream Tasks"***, accepted in the Adversarial Machine Learning on Computer Vision: Foundation Models + X (**ADVML**) Workshop of **CVPR 2025**. 

> [**# Attacking Attention of Foundation Models Effectively Disrupts Downstream Tasks"**](https://arxiv.org/abs/2506.05394) <br>Hondamunige Prasanna Silva, Federico Becattini and  Lorenzo Seidenari<br>
> 
> **Abstract:** Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without finetuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT) , are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems,jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation



## Table of Contents
- [Abstract](#abstract)
- [Installation](#install-conda-env)
- [Project Structure](#project-structure)
- [Download Models & Datasets](#download-the-model-and-datasets)
- [How to Launch the Attack](#how-to-lunch-sga-attack)
- [Usage](#how-to-use)
- [Evaluation](#attack-evaluation)
- [TODO](#todo)
- [Citation](#citation)


## Install conda env
```
conda env create -f environment.yml
```

## Run the project as a Packages 
```bash
pip install .  
pip instlal -e .  # If you want in edit mode!
```


## Project Structure

```
attack-attention
├─ .gitignore
├─ .pre-commit-config.yaml
├─ LICENSE
├─ README.md
├─ checkpoint
│  ├─ albef
│  └─ tcl
├─ data
│  ├─ flickr30k-images
│  └─ val2014
├─ SGA
│  └─ ...
├─ data_annotation
│  ├─ coco_test.json
│  └─ flickr30k_test.json
├─ environment.yml
├─ eval_clip-vit2albef.py
├─ eval_clip-vit2clip-cnn.py
├─ eval_clip.py
├─ models
│  ├─ __init__.py
│  ├─ clip_model
│  │  ├─ model.py
│  │  └─ ..
│  └─ ..
└─ std_eval_idx
   ├─ flickr30k
   └─ mscoco

```
## Download the model and datasets
Create `checkpoint` folder inside the root folder and download TCL and ALBEF models.
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL).

Create `data` folder inside the root folder and download [FLICKR30K](https://drive.google.com/file/d/1lyvBZVd6nHCoEq9_ssqQnEqskAi_nTlJ/view?usp=sharing), [MSCOCO](https://drive.google.com/file/d/1_gu9ycXB-unY_g4gDjLVkbPlKIhxgGlU/view?usp=sharing).

## How to lunch SGA attack
Check readme.md inside `SGA folder`.
# How to use

## Parameters
| Argument | Default | Type | Description |
|----------|---------|------|-------------|
| `--config` | `./configs/Retrieval_flickr.yaml` | `str` | Path to the configuration YAML file. |
| `--seed` | `42` | `int` | Seed for random number generation to ensure reproducibility. |
| `--source_model` | `ViT-B/16` | `str` | Name of the source model used for evaluation. |
| `--source_text_encoder` | `bert-base-uncased` | `str` | Text encoder architecture for the source model. |
| `--source_ckpt` | `None` | `str` | Path to the checkpoint for the source model, if any. |
| `--target_model` | `ALBEF` | `str` | Target model for evaluation. |
| `--target_text_encoder` | `bert-base-uncased` | `str` | Text encoder for the target model. |
| `--target_ckpt` | `./checkpoint/albef/flickr30k.pth` | `str` | Path to the checkpoint for the target model. |
| `--original_rank_index_path` | `./std_eval_idx/flickr30k/` | `str` | Path to the file containing original rank indices. |
| `--alpha` | `90` | `int` | Weight for the loss function during adversarial optimization. |
| `--loss` | `atn` | `str` | Type of loss function to use. |
| `--method` | `1` | `int` | Identifier for the attack method; refer to the `my_attack` function for details. |
| `--index` | `11` | `int` | Layer index to attack in the target model. |


## Attack evaluation
To eval the attack `without trasferability`
```python
python eval_clip.py  --config ./configs/Retrieval_coco.yaml \
   --source_model ViT-B/16 --original_rank_index ./std_eval_idx/mscoco/ \
   --loss atn --method 1;
```

To eval the attack with trasferability (`ALBEF`)
```python
python eval_clip-vit2albef.py  --config ./configs/Retrieval_flickr.yaml \
   --source_model ViT-B/16  --target_model ALBEF --target_ckpt ./checkpoint/albef/flickr30k.pth \
   --original_rank_index ./std_eval_idx/flickr30k/ --loss atn --method 1
```

To eval the attack with trasferability (`TCL`)
```python
python eval_clip-vit2albef.py --config ./configs/Retrieval_flickr.yaml \
   --source_model ViT-B/16 --target_model TCL --target_ckpt ./checkpoint/tcl/checkpoint_flickr_finetune.pth \
   --original_rank_index ./std_eval_idx/flickr30k/ --loss atn --method 1
```

To eval the attack with trasferability (`CLIP-CNN`)
```python
python eval_clip-vit2clip-cnn.py --config ./configs/Retrieval_flickr.yaml \
--source_model ViT-B/16  --target_model RN101 \
--original_rank_index ./std_eval_idx/flickr30k/ --loss atn --method 1
```

#### Parameter used for experiments
`--source_model ViT-B/16 --method 3 --loss emb`


## TODO
- [ ] Complete the repo with all the downstream task used during experiments
- [ ] Refactor the code

## Citation

If you find this work useful, please consider citing it:
```bibtex
@article{silvaattacking,
  title={Attacking Attention of Foundation Models Disrupts Downstream Tasks},
  author={Silva, Hondamunige Prasanna and Becattini, Federico and Seidenari, Lorenzo}
}
```
