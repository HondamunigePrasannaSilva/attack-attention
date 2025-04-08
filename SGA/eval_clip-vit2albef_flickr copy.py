import argparse

# import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime

import torch
import yaml
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import BertForMaskedLM
from torchvision import transforms
from PIL import Image

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import clip

import SGA.utils as utils

from attacker import SGAttacker, ImageAttacker, TextAttacker

from dataset import paired_dataset


def retrieval_eval(
    model,
    ref_model,
    t_model,
    t_ref_model,
    t_test_transform,
    data_loader,
    tokenizer,
    t_tokenizer,
    device,
    config,
):
    # test
    model.float()
    model.eval()
    ref_model.eval()
    t_model.float()
    t_model.eval()
    t_ref_model.eval()

    print("Computing features for evaluation adv...")

    images_normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    img_attacker = ImageAttacker(
        images_normalize, eps=2 / 255, steps=10, step_size=0.5 / 255
    )
    txt_attacker = TextAttacker(
        ref_model,
        tokenizer,
        cls=False,
        max_length=77,
        number_perturbation=1,
        topk=10,
        threshold_pred_score=0.3,
    )
    attacker = SGAttacker(model, img_attacker, txt_attacker)


    if args.scales is not None:
        scales = [float(itm) for itm in args.scales.split(",")]
        print(scales)
    else:
        scales = None

    print("Forward")
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(
        data_loader
    ):
        print(f"--------------------> batch:{batch_idx}/{len(data_loader)}")
        texts_ids = []
        txt2img = []
        texts = []
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i] * len(text_ids_groups[i])

        images = images.to(device)
        adv_images, adv_texts = attacker.attack(
            images, texts, txt2img, device=device, max_lemgth=77, scales=scales
        )
        




def load_model(model_name, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)

    model, preprocess = clip.load(model_name, device=device)
    model.set_tokenizer(tokenizer)
    return model, ref_model, tokenizer




def main(args, config):
    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating Source Model")
    model, ref_model, tokenizer = load_model(
        args.source_model, args.source_ckpt, args.source_text_encoder, device
    )
    t_model, t_ref_model, t_tokenizer = load_model(
        args.target_model, args.target_ckpt, args.target_text_encoder, device
    )

    #### Dataset ####
    print("Creating dataset")
    n_px = model.visual.input_resolution

    print(n_px)
    s_test_transform = transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=Image.BICUBIC, antialias=True),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
        ]
    )

    test_dataset = paired_dataset(
        config["test_file"], s_test_transform, config["image_root"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_coco.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    parser.add_argument("--source_model", default="ViT-B/16", type=str)
    parser.add_argument("--source_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument("--source_ckpt", default=None, type=str)

    parser.add_argument("--target_model", default="ALBEF", type=str)
    parser.add_argument("--target_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--target_ckpt", default="../checkpoint/albef/flickr30k.pth", type=str
    )

    parser.add_argument(
        "--original_rank_index_path", default="../std_eval_idx/flickr30k/"
    )
    parser.add_argument("--scales", type=str, default="0.5,0.75,1.25,1.5")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    main(args, config)
