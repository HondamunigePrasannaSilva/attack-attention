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
from models.clip_model import clip

import SGA.utils as utils

from SGA.dataset import paired_dataset

import torch.nn.functional as F

    
def my_attack(model, image, args):
    patch_size = 224
    adv_patch = torch.randint(
        0,
        1,
        size=(3, patch_size, patch_size),
        dtype=torch.float32,
    )

    adv_patch.data = torch.rand((3, patch_size, patch_size))
    adv_patch.data = (adv_patch.data - 0.5) / 0.5
    # set requires_grad to True
    adv_patch.requires_grad = True
    optimizer = torch.optim.AdamW([adv_patch], lr=0.01)
    eps_ = 2 / 255
    eps = torch.tensor([eps_ / 0.26862954, eps_ / 0.26130258, eps_ / 0.27577711])
    
    
    with torch.no_grad():
        image_features, atn, my_x, v_clean = model.encode_image(image)

    index = args.index

    if args.loss == "atn" and args.method == 1:
        v_clean = v_clean[index].squeeze()
        
    if args.loss == "atn" and args.method == 2 or args.method == 3:
        atn = atn[index].squeeze()
        

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    lower_bound = (0 - mean) / std
    upper_bound = (1 - mean) / std

    for i in range(250):
        x_adv = image.cpu().clone()
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        x_adv = x_adv + adv_patch
        adv_feature, attn, my_adv_x, adv_v = model.encode_image(x_adv.cuda())

        if args.loss == "all":
            loss = 30 * (attn[:, 1:, 1:] * atn[:, 1:, 1:]).mean(0).mean(
                1
            ).sum() + 30 * ((my_adv_x.squeeze()[0] * my_x.squeeze()[0]).mean())
        elif args.loss == "atn":
            if args.method == 1:  # attacking the value matrix
                adv_v = adv_v[index].squeeze()

                loss = (adv_v[:, 0, :] * v_clean[:, 0, :]).mean(0).sum()
            # only cls
            if args.method == 2:
                attn = attn[index].squeeze()
                loss = args.alpha * (attn[:, 0, :] * atn[:, 0, :]).mean(0).sum()
            # only visual
            if args.method == 3:
                attn = attn[index].squeeze()
                loss = (
                        args.alpha
                        * (attn[:, 1:, 1:] * atn[:, 1:, 1:]).mean(0).mean(1).sum()
                    )

        else:  # embdding loss
            if args.method == 1:  # #embedding pre projection
                loss = (my_adv_x.squeeze()[0] * my_x.squeeze()[0]).mean()
            if args.method == 2:  # final embeddings
                loss = (adv_feature.squeeze() * image_features.squeeze()).mean()
            if args.method == 3:
                attn = attn[index].squeeze()
                loss_1 = -torch.norm(adv_feature.squeeze() - image_features.squeeze(), p=2) 
                loss = loss_1+ args.alpha * (attn[:, 1:, 1:] * atn[:, 1:, 1:]).mean(0).mean(1).sum()
            if args.method == 4:
                loss = -torch.norm(adv_feature.squeeze() - image_features.squeeze(), p=2)
                

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for c in range(adv_patch.shape[0]):  # Iterate over channels
                adv_patch[c].data.clamp_(min=-eps[c].item(), max=eps[c].item())

    _image = image[0].cpu().clone()
    adv_img = _image + adv_patch

    return adv_feature, torch.clamp(adv_img, lower_bound, upper_bound)


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
    args,
):
    # test
    model.float()
    model.eval()
    ref_model.eval()
    t_model.float()
    t_model.eval()
    t_ref_model.eval()

    print("Computing features for evaluation adv...")

    print("Prepare memory")
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    s_image_feats = torch.zeros(num_image, model.visual.output_dim)
    s_text_feats = torch.zeros(num_text, model.visual.output_dim)

    t_image_feats = torch.zeros(num_image, config["embed_dim"])
    t_image_embeds = torch.zeros(num_image, 577, 768)
    t_text_feats = torch.zeros(num_text, config["embed_dim"])
    t_text_embeds = torch.zeros(num_text, 30, 768)
    t_text_atts = torch.zeros(num_text, 30).long()

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

        # adv_images, adv_texts = attacker.attack(images, texts, txt2img, device=device,
        # max_lemgth=77, scales=scales)
        adv_fe, adv_images = my_attack(model, images, args)

        adv_texts = texts
        adv_images = adv_images.cuda()

        with torch.no_grad():
            output = model.inference(adv_images, adv_texts)
            s_image_feats[images_ids] = output["image_feat"].cpu().float().detach()
            s_text_feats[texts_ids] = output["text_feat"].cpu().float().detach()

            t_adv_img_list = []
            for itm in adv_images:
                t_adv_img_list.append(t_test_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)

            t_adv_images_norm = t_adv_imgs  # images_normalize(t_adv_imgs)
            adv_texts_input = tokenizer(
                adv_texts,
                padding="max_length",
                truncation=True,
                max_length=30,
                return_tensors="pt",
            ).to(device)
            t_output_img = t_model.inference_image(t_adv_images_norm)
            t_output_txt = t_model.inference_text(adv_texts_input)
            t_image_feats[images_ids] = t_output_img["image_feat"].cpu().detach()
            t_image_embeds[images_ids] = t_output_img["image_embed"].cpu().detach()
            t_text_feats[texts_ids] = t_output_txt["text_feat"].cpu().detach()
            t_text_embeds[texts_ids] = t_output_txt["text_embed"].cpu().detach()
            t_text_atts[texts_ids] = adv_texts_input.attention_mask.cpu().detach()

    s_sims_matrix = s_image_feats @ s_text_feats.t()

    t_score_matrix_i2t, t_score_matrix_t2i = retrieval_score(
        t_model,
        t_image_feats,
        t_image_embeds,
        t_text_feats,
        t_text_embeds,
        t_text_atts,
        num_image,
        num_text,
        device=device,
    )

    return (
        s_sims_matrix.cpu().numpy(),
        s_sims_matrix.t().cpu().numpy(),
        t_score_matrix_i2t.cpu().numpy(),
        t_score_matrix_t2i.cpu().numpy(),
    )


@torch.no_grad()
def retrieval_score(
    model,
    image_feats,
    image_embeds,
    text_feats,
    text_embeds,
    text_atts,
    num_image,
    num_text,
    device=None,
):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation Direction Similarity With Bert Attack:"

    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)

        encoder_output = image_embeds[i].repeat(config["k_test"], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            device
        )
        output = model.text_encoder(
            encoder_embeds=text_embeds[topk_idx].to(device),
            attention_mask=text_atts[topk_idx].to(device),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode="fusion",
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
            device
        )
        output = model.text_encoder(
            encoder_embeds=text_embeds[i].repeat(config["k_test"], 1, 1).to(device),
            attention_mask=text_atts[i].repeat(config["k_test"], 1).to(device),
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_att,
            return_dict=True,
            mode="fusion",
        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]

    original_rank_index_path = args.original_rank_index_path
    origin_tr1 = np.load(f"{original_rank_index_path}/{model_name}_tr1_rank_index.npy")
    origin_tr5 = np.load(f"{original_rank_index_path}/{model_name}_tr5_rank_index.npy")
    origin_tr10 = np.load(
        f"{original_rank_index_path}/{model_name}_tr10_rank_index.npy"
    )

    asr_tr1 = round(
        100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2
    )  
    asr_tr5 = round(
        100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2
    )
    asr_tr10 = round(
        100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2
    )

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f"{original_rank_index_path}/{model_name}_ir1_rank_index.npy")
    origin_ir5 = np.load(f"{original_rank_index_path}/{model_name}_ir5_rank_index.npy")
    origin_ir10 = np.load(
        f"{original_rank_index_path}/{model_name}_ir10_rank_index.npy"
    )

    asr_ir1 = round(
        100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2
    )
    asr_ir5 = round(
        100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2
    )
    asr_ir10 = round(
        100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2
    )

    eval_result = {
        "txt_r1_ASR (txt_r1)": f"{asr_tr1}({tr1})",
        "txt_r5_ASR (txt_r5)": f"{asr_tr5}({tr5})",
        "txt_r10_ASR (txt_r10)": f"{asr_tr10}({tr10})",
        "img_r1_ASR (img_r1)": f"{asr_ir1}({ir1})",
        "img_r5_ASR (img_r5)": f"{asr_ir5}({ir5})",
        "img_r10_ASR (img_r10)": f"{asr_ir10}({ir10})",
    }
    return eval_result


def load_model(model_name, model_ckpt, text_encoder, device):
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    ref_model = BertForMaskedLM.from_pretrained(text_encoder)
    if model_name in ["ALBEF", "TCL"]:
        model = ALBEF(config=config, text_encoder=text_encoder, tokenizer=tokenizer)
        checkpoint = torch.load(model_ckpt, map_location="cpu")
    ### load checkpoint
    else:
        model, preprocess = clip.load(model_name, device=device)
        model.set_tokenizer(tokenizer)
        return model, ref_model, tokenizer

    try:
        state_dict = checkpoint["model"]
    except Exception:
        state_dict = checkpoint

    if model_name == "TCL":
        pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder.pos_embed"], model.visual_encoder
        )
        state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(
            state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
        )
        state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped

    for key in list(state_dict.keys()):
        if "bert" in key:
            encoder_key = key.replace("bert.", "")
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model, ref_model, tokenizer


def eval_asr(
    model,
    ref_model,
    tokenizer,
    t_model,
    t_ref_model,
    t_tokenizer,
    t_test_transform,
    data_loader,
    device,
    args,
    config,
):
    model = model.to(device)
    ref_model = ref_model.to(device)

    t_model = t_model.to(device)
    t_ref_model = t_ref_model.to(device)

    print("Start eval")
    start_time = time.time()

    score_i2t, score_t2i, t_score_i2t, t_score_t2i = retrieval_eval(
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
        args,
    )

    t_result = itm_eval(
        t_score_i2t,
        t_score_t2i,
        data_loader.dataset.img2txt,
        data_loader.dataset.txt2img,
        "ALBEF",
    )
    print("Performance on {}: \n {}".format(args.target_model, t_result))

    result = itm_eval(
        score_i2t,
        score_t2i,
        data_loader.dataset.img2txt,
        data_loader.dataset.txt2img,
        "CLIP_ViT",
    )

    print("Performance on {}: \n {}".format(args.source_model, result))
    print("Performance on {}: \n {}".format(args.target_model, t_result))
    # write the results in a text file

    output_file = "performance_results.txt"

    with open(output_file, "a") as file:
        file.write(f"\n target model: {args.target_model}, loss {args.loss} \n")
        file.write("Performance on {}: \n{}\n".format(args.source_model, result))
        file.write("Performance on {}: \n{}\n".format(args.target_model, t_result))

    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluate time {}".format(total_time_str))


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
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    t_test_transform = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.Resize(
                (config["image_res"], config["image_res"]),
                interpolation=Image.BICUBIC,
                antialias=True,
            ),
            # transforms.ToTensor(),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
    )

    test_dataset = paired_dataset(
        config["test_file"], s_test_transform, config["image_root"]
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=4,
        collate_fn=test_dataset.collate_fn,
    )

    eval_asr(
        model,
        ref_model,
        tokenizer,
        t_model,
        t_ref_model,
        t_tokenizer,
        t_test_transform,
        test_loader,
        device,
        args,
        config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_flickr.yaml")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--source_model", default="ViT-B/16", type=str)
    parser.add_argument("--source_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument("--source_ckpt", default=None, type=str)

    parser.add_argument("--target_model", default="ALBEF", type=str)
    parser.add_argument("--target_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument(
        "--target_ckpt", default="./checkpoint/albef/flickr30k.pth", type=str
    )

    parser.add_argument(
        "--original_rank_index_path", default="./std_eval_idx/flickr30k/"
    )

    parser.add_argument("--alpha", type=int, default=90)  # weight for the loss
    parser.add_argument("--loss", type=str, default="atn")  # loss type
    parser.add_argument("--method", type=int, default=1)  # see my_attack function
    parser.add_argument("--index", type=int, default=11)  # layer index to attack
    parser.add_argument("--resize", type=int, default=0)  # layer index to attack
    

    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    main(args, config)
