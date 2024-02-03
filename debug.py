import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F 
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import load_checkpoint
import pandas as pd

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset, build_dataloader
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def parse_args():
    parser = argparse.ArgumentParser(description='Test a well-trained model')
    parser.add_argument(
        '--config', default = 'model_pth/visece/pacldebug1/test_pacl.py',help='train config file path'
    )
    parser.add_argument(
        '--checkpoint',
        default='model_pth/visece/pacldebug1/epoch_24.pth',
        help='pretrained model path'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.checkpoint:
        cfg.setdefault('checkpoint', args.checkpoint)
    else:
        raise ValueError('plz provide the model checkpoint')
    train_dataset = build_dataset(
        cfg.data.train
    )
    val_dataset = build_dataset(
        cfg.data.val
    )
    dataloader_setting = dict(
        videos_per_gpu = cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        shuffle = True
    )
    train_loader = build_dataloader(
        train_dataset, seed = 0, **dataloader_setting
    )
    model = build_model(
        cfg.model
    )
    load_checkpoint(model, cfg.checkpoint, map_location='cpu')
    classes_all = pd.read_csv(
        cfg.model.text_lists
    ).values
    # 测试训练集的识别结果
    device = torch.device('cuda:0')
    model.to(device)
    for samples in train_loader:
        imgs = samples['imgs'].to(device)
        labels = samples['label']
        texts = []
        for idx in labels:
            texts.append(classes_all[idx][1])
        # 训练流程的前向过程
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        video_embeds = model.video_encoder(imgs) # [16, 24, 512]
        video_feat_raw = model.video_embed_linear(video_embeds) + model.video_embed_res(video_embeds) # [16, 24, 256]
        text = model.tokenizer(
            texts, padding = 'max_length', truncation = True, 
            max_length = 10, return_tensors = 'pt'
        ).to(device)
        # text.input_ids [16, 10]
        text_output = model.text_encoder(
            text.input_ids, attention_mask = text.attention_mask,
            return_dict = True, mode = 'text'
        ) # .last_hidden_state [16, 10, 768]
        # PACL论文
        # 先计算et
        text_feat_raw = model.text_proj(
            text_output.last_hidden_state[:,0,:]
        ) # [16, 256]
        # 通过patch alignment得到融合后的video_feat
        video_feat = model.patch_alignment_train_mod(video_feat_raw, text_feat_raw)
        # 归一化
        video_feat = F.normalize(video_feat, dim=-1) # [16, 256]
        text_feat = F.normalize(text_feat_raw, dim=-1) # [16, 256]
        # 验证流程的前向过程
        all_texts = []
        for i in range(len(classes_all)):
            all_texts.append(classes_all[i][1])
        all_text = model.tokenizer(
            all_texts, padding = 'max_length', truncation=True,
            max_length = 10, return_tensors = 'pt'
        ).to(device)
        all_text_output = model.text_encoder(
            all_text.input_ids, attention_mask = all_text.attention_mask,
            return_dict = True, mode = 'text'
        )
        all_text_feat_raw = model.text_proj(all_text_output.last_hidden_state[:, 0, :])
        #'''
        # 按照验证流程测试负样本对最终的相似度
        v8_raw = video_feat_raw[0] # [T, C] 视频类别8
        norm_v8_raw = F.normalize(v8_raw, dim=-1) # 归一化
        t0_raw = all_text_feat_raw[0] # [C] 文本类别0
        norm_t0_raw = F.normalize(t0_raw, dim=-1) # 归一化
        a_v8_t0 = norm_v8_raw @ norm_t0_raw.t() # 计算相似度
        a_v8_t0_sig = F.sigmoid(a_v8_t0 * 10) # [T]
        mod_v8 = torch.sum(a_v8_t0_sig.unsqueeze(-1) * v8_raw, dim=0) # 加权后的视频特征
        simi_v1 = mod_v8 @ norm_t0_raw.t()
        simi_v2 = F.normalize(mod_v8, dim=-1) @ norm_t0_raw.t()
        pass
        
        # 我的视频和任意的文本计算相似度
        my_texts = ['I hate this code']
        my_text = model.tokenizer(
            my_texts, padding = 'max_length', truncation = True, 
            max_length = 10, return_tensors = 'pt'
        ).to(device)
        my_text_output = model.text_encoder(
            my_text.input_ids, attention_mask = my_text.attention_mask,
            return_dict = True, mode = 'text'
        )
        my_text_feat_raw = model.text_proj(
            my_text_output.last_hidden_state[:,0,:]
        ) 
        v8_raw = video_feat_raw[0] # [T, C] 视频类别8
        norm_v8_raw = F.normalize(v8_raw, dim=-1) # 归一化
        my_text_raw = my_text_feat_raw[0] # 我的随意设的文本
        norm_my_text_raw = F.normalize(my_text_raw, dim=-1) # 归一化
        a_v8_tm = norm_v8_raw @ norm_my_text_raw.t() # 计算相似度
        a_v8_tm_sig = F.sigmoid(a_v8_tm*10) # [T]
        mod_v8 = torch.sum(a_v8_tm_sig.unsqueeze(-1) * v8_raw, dim=0) # 加权后的视频特征
        sim = F.normalize(mod_v8, dim=-1) @ norm_my_text_raw.t()
        pass
        
        pass


if __name__ == '__main__':
    main()