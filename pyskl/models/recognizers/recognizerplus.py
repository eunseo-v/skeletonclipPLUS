import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from .. import builder
from ..builder import RECOGNIZERS

from transformers import BertTokenizer
import pandas as pd

@RECOGNIZERS.register_module()
class RecognizerPlus(nn.Module, metaclass = ABCMeta):
    '''
    Methods:
        forward_train: supporting to forward when training
        forward_test: supporting to forward when testing
    包含video_encoder和text_encoder
    video_encoder就是正常的PoseConv3D
    text_encoder基于BertModel，输出文本特征
    '''
    def __init__(
            self, video_encoder, text_encoder, embed_dim, itm_dim,
            frozen_bert = False,
            text_encoder_init = 'bert-base-uncased', text_lists = 'lists/ntu_60_labels.csv'
    ):
        super().__init__()
        # 构建video_encoder和text_encoder
        self.video_encoder = builder.build_backbone(video_encoder)
        self.text_encoder = builder.build_backbone(text_encoder)
        self.vision_proj = nn.Linear(
            video_encoder.res_out_channels,
            embed_dim
        )
        self.text_proj = nn.Linear(
            text_encoder.config.hidden_size,
            embed_dim
        )
        self.itm_head = nn.Linear(itm_dim, 2) # [768, 2]
        self.frozen_bert = frozen_bert
        self.text_encoder_init = text_encoder_init
        self.classes_all = pd.read_csv(text_lists).values
        self.tokenizer = self.init_tokenizer()
        self.temp = nn.Parameter(0.07*torch.ones([]))
        self.init_weights()

    def init_tokenizer(self):
        # BertTokenizer常用用法
        # from_pretrained: 从包含词表文件的目录中初始化一个分词器
        # tokenize: 将文本(词或者句子)分解为子词列表
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # uncased的不区分大小写，需要事先lowercase
        tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer
    
    def init_weights(self):
        # video_encoder就是poseconv3d，采用其正常的初始化策略
        self.video_encoder.init_weights()
        self.text_encoder.from_pretrained(self.text_encoder_init)
        if self.frozen_bert:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def patch_alignment_train(self, video_feat_raw, text_feat_raw):
        # 首先归一化
        video_feat_raw = F.normalize(video_feat_raw, dim=-1)
        text_feat_raw = F.normalize(text_feat_raw, dim=-1)
        # 计算frame level similarity
        s_ev_et = torch.einsum('btc,bc->bt', video_feat_raw, text_feat_raw) # [B, T]
        # softmax
        a_ev_et = F.softmax(s_ev_et, dim=-1) # [B, T]
        # 加权后的视频特征
        video_feat = torch.einsum('bt,btc->bc', a_ev_et, video_feat_raw) # [B, C=256]
        return video_feat
    
    def patch_alignment_val(self, video_feat_raw, text_feat_raw):
        # video_feat_raw [B, T, C]
        # text_feat_raw [L, C]
        # 首先归一化
        video_feat_raw = F.normalize(video_feat_raw, dim=-1)
        text_feat_raw = F.normalize(text_feat_raw, dim=-1)
        # 计算每一个样本的每一帧对于所有L类标签文本的相似度
        s_ev_et = torch.einsum('btc,lc->blt', video_feat_raw, text_feat_raw) # [B, L, T]
        # a_ev_et = F.softmax(s_ev_et, dim=-1) # [B, L, T]
        a_ev_et = F.sigmoid(s_ev_et*10)
        video_feat = torch.einsum('blt,btc->blc', a_ev_et, video_feat_raw) # [B, L, C]
        return video_feat
    
    def patch_alignment_train_mod(self, video_feat_raw, text_feat_raw):
        # video_feat_raw [B, T, C]
        # text_feat_raw [B, C]
        # normalize归一化
        normalized_video_feat_raw = F.normalize(video_feat_raw, dim=-1) # [B, T, C]
        normalized_video_feat_raw = normalized_video_feat_raw.transpose(-2, -1) # [B, C, T]
        normalized_text_feat_raw = F.normalize(text_feat_raw, dim=-1) # [B, C]
        normalized_text_feat_raw = normalized_text_feat_raw.unsqueeze(1) # [B, 1, C]
        # 计算点乘
        activations = normalized_text_feat_raw @ normalized_video_feat_raw # [B, 1, T]
        activations = activations.squeeze() # [B, T]
        # 由于是点乘，范围是-1(least similar)到+1(most similar)
        # multiply by 10 and apply sigmoid function. 
        # this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        # activations = F.sigmoid(activations*10) # [B, T]
        activations = F.softmax(activations, dim=-1)
        video_feat = torch.sum(
            video_feat_raw* (activations.unsqueeze(-1)), dim=1
        )
        return video_feat
    
    def patch_alignment_val_mod(self, video_feat_raw, text_feat_raw):
        # video_feat_raw [B, T, C]
        # text_feat_raw [L, C]
        # normalize归一化
        normalized_video_feat_raw = F.normalize(video_feat_raw, dim=-1) # [B, T, C]
        normalized_video_feat_raw = normalized_video_feat_raw.transpose(-2, -1) # [B, C, T]
        normalized_text_feat_raw = F.normalize(text_feat_raw, dim=-1) # [L, C]
        # 计算点乘
        activations = torch.einsum('bct,lc->blt', normalized_video_feat_raw, normalized_text_feat_raw)
        activations = F.sigmoid(activations*10) # [B, L, T]
        # activations = F.softmax(activations, dim=-1)
        video_feat = torch.einsum('blt,btc->blc', activations, video_feat_raw) # [B, L, C]
        return video_feat
    '''
    def forward_train(self, imgs, label, **kwargs):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        device = imgs.device
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        texts = []
        for idx in label:
            texts.append(self.classes_all[idx][1])
        # 前向过程
        # video_encoder
        video_embeds = self.video_encoder(imgs) # [B, 48, 512]
        # video embedder：残差块结构
        video_feat_raw = self.video_embed_linear(video_embeds) + self.video_embed_res(video_embeds) # [B, T, 256]
        
        # text_encoder
        text = self.tokenizer(
            texts, padding = 'max_length', truncation = True,
            max_length = 10, return_tensors = 'pt'
        ).to(device)    # 这时候的起始token为CLS
        # text.input_ids [B, 10]
        text_output = self.text_encoder(
            text.input_ids, attention_mask = text.attention_mask,
            return_dict = True, mode = 'text'
        ) # .last_hidden_state [B, 10, 768]
        # PACL论文中最后阶段才进行归一化
        # 先计算et
        text_feat_raw = self.text_proj(
            text_output.last_hidden_state[:,0,:]
        ) # [B, 256]
        # 通过patch alignment得到融合后的video feat
        video_feat = self.patch_alignment_train(video_feat_raw, text_feat_raw)
        # video_feat = self.patch_alignment_train_mod(video_feat_raw, text_feat_raw)
        # 归一化
        # video_feat = F.normalize(video_feat, dim=-1) # [B, 256]
        text_feat = F.normalize(text_feat_raw, dim=-1)  # [B, 256]
        # 计算真值 sim_targets [B, B] 行和为1
        label = label.view(-1, 1) # [B, 1]
        pos_idx = torch.eq(label, label.t()).float() # [B, B]
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True) # 归一化
        # 计算两种相似度
        sim_i2t = video_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ video_feat.t() / self.temp
        # 计算loss
        # cross_entropy=-pi·log(qi)可以分解成三部分
        # 1) 对sim_i2t求softmax得到qi再进行log -> log_softmax
        # 2) -pi·log(qi)
        # 3) 相加取平均 torch.sum().mean()
        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1)*sim_targets,
            dim=1
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1)*sim_targets,
            dim=1
        ).mean()
        losses = dict()
        losses['loss_cls'] = (loss_i2t + loss_t2i)/2

        return losses
    '''

    # 按照forward_train 按照验证的流程，计算与所有文本的帧特征
    def forward_train(self, imgs, label, **kwargs):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        device = imgs.device
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) 
        texts = []
        for i in range(len(self.classes_all)):
            texts.append(self.classes_all[i][1])
        text = self.tokenizer(
            texts, padding = 'max_length', truncation = True,
            max_length = 10, return_tensors = 'pt'
        ).to(device)
        # text.input_ids [60, 10]
        text_output = self.text_encoder(
            text.input_ids, attention_mask = text.attention_mask,
            return_dict = True, mode = 'text'
        ) # .last_hidden_state [60, 10, 768]
        text_feat_raw = self.text_proj(text_output.last_hidden_state[:,0,:]) # [60, 256]
        # 计算得到imgs对应的video_feat_raw
        video_embeds = self.video_encoder(imgs) # [B, T, 512]
        # video embedder：残差块结构
        # video_feat_raw = self.video_embed_linear(video_embeds) + self.video_embed_res(video_embeds) # [B, T, 256]
        # visece的embedder结构
        video_feat_raw = self.vision_proj(video_embeds)
        # 方案1：使用softmax计算权重，作用于归一化的帧向量
        # video_feat = self.patch_alignment_val(video_feat_raw, text_feat_raw) # [B, L, C]
        video_feat = self.patch_alignment_val_mod(video_feat_raw, text_feat_raw)
        # 归一化
        video_feat = F.normalize(video_feat, dim=-1) # [B, L, 256]
        text_feat = F.normalize(text_feat_raw, dim=-1) # [L, 256]
        # 点乘计算相似度
        sim_i2t = torch.einsum('blc, lc->bl', video_feat, text_feat)/self.temp # [B, L]
        # 计算真值
        sim_targets = torch.zeros_like(sim_i2t)
        for i, idx in enumerate(label):
            sim_targets[i][idx] = 1.0
        # 计算loss
        # cross_entropy = -pi·log(qi)可以分解成三部分
        # 1) 对sim_i2t求softmax得到qi再进行log -> log_softmax
        # 2) -pi·log(qi)
        # 3) 相加取平均 torch.sum().mean()
        loss = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_targets,
            dim=1
        ).mean()
        # itm 前向过程

        losses = dict()
        losses['loss_cls'] = loss
        return losses

    # forward_test得到的是每一个样本分割成num_seg份后，对于每一类的可能性
    # 输出cls_score [B, num_classes] 行和为1
    def forward_test(self, imgs, **kwargs):
        # 60类文本的text_feats [60, 256]
        device = imgs.device
        texts = []
        for i in range(len(self.classes_all)):
            texts.append(self.classes_all[i][1])
        text = self.tokenizer(
            texts, padding = 'max_length', truncation = True,
            max_length = 10, return_tensors = 'pt'
        ).to(device)
        # text.input_ids [60, 10]
        text_output = self.text_encoder(
            text.input_ids, attention_mask = text.attention_mask,
            return_dict = True, mode = 'text'
        ) # .last_hidden_state [60, 10, 768]
        text_feat_raw = self.text_proj(text_output.last_hidden_state[:,0,:]) # [60, 256]
        # 计算得到imgs对应的video_feat_raw
        # imgs [B, num_segs, C, T, H, W]
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # [B*num_sugs, C, T, H, W]
        video_embeds = self.video_encoder(imgs) # [B, 48, 512]
        # video embedder：残差块结构
        # video_feat_raw = self.video_embed_linear(video_embeds) + self.video_embed_res(video_embeds) # [B, T, 256]
        # visece的embedder结构
        video_feat_raw = self.vision_proj(video_embeds)
        # 得到通过patch alignment后的视频特征
        # video_feat = self.patch_alignment_val(video_feat_raw, text_feat_raw)
        video_feat = self.patch_alignment_val_mod(video_feat_raw, text_feat_raw)
        # 归一化
        video_feat = F.normalize(video_feat, dim=-1) # [B, L, 256]
        text_feat = F.normalize(text_feat_raw, dim=-1)  # [L, 256]
        # 点乘计算相似度
        cls_score = torch.einsum('blc, lc->bl', video_feat, text_feat) # [B*num_segs, L]
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        cls_score = F.softmax(cls_score, dim=2).mean(dim=1) # [B, 60]
        return cls_score.cpu().numpy()

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, label, **kwargs)
        # 验证的时候会进入
        return self.forward_test(imgs, **kwargs)
    
    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch, return_loss=True)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        return outputs
    
    def _parse_losses(self, losses):
        """Parse the ra w outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars    
    
    def is_same_class(self, labels):
        return torch.allclose(labels, labels[0])