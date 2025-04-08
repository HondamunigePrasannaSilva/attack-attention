# Set-level Guidance Attack


### 2. Prepare datasets and models
The checkpoints of the fine-tuned VLP models is accessible in [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL)

### 3. Attack evaluation
From ALBEF to TCL on the Flickr30k dataset:
```python
python eval_albef2tcl_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ALBEF  --source_ckpt ../checkpoint/flickr30k.pth \
--target_model TCL --target_ckpt ./checkpoint/checkpoint_flickr_finetune.pth \
--original_rank_index ../std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

From ALBEF to CLIP<sub>ViT</sub> on the Flickr30k dataset:
```python
python eval_albef2clip-vit_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ALBEF  --source_ckpt ../checkpoint/flickr30k.pth \
--target_model ViT-B/16 --original_rank_index ../std_eval_idx/flickr30k/ \
--scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to ALBEF on the Flickr30k dataset:
```python
python eval_clip-vit2albef_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ViT-B/16  --target_model ALBEF \
--target_ckpt ../checkpoint/flickr30k.pth \
--original_rank_index ../std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

From CLIP<sub>ViT</sub> to CLIP<sub>CNN</sub> on the Flickr30k dataset:
```python
python eval_clip-vit2clip-cnn_flickr.py --config ./configs/Retrieval_flickr.yaml \
--source_model ViT-B/16  --target_model RN101 \
--original_rank_index ../std_eval_idx/flickr30k/ --scales 0.5,0.75,1.25,1.5
```

### Citation
Kindly include a reference to this paper in your publications if it helps your research:
```
@misc{lu2023setlevel,
    title={Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models},
    author={Dong Lu and Zhiqiang Wang and Teng Wang and Weili Guan and Hongchang Gao and Feng Zheng},
    year={2023},
    eprint={2307.14061},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
