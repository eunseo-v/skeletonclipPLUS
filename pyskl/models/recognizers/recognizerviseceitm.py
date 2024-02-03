import torch.nn as nn
from ..builder import RECOGNIZERS
from .recognizervisece import RecognizerVisece

import torch
import torch.nn.functional as F

@RECOGNIZERS.register_module()
class RecognizerViseceITM(RecognizerVisece):
    def __init__(
            self, itm_dim, **kwargs
    ):
        super().__init__(**kwargs)
        self.itm_head = nn.Linear(itm_dim, 2) # [768, 2]

    def forward_train(self, imgs, label, **kwargs):
        # 我需要添加一个逻辑，如果一个batch中的标签属于同一类，就不去计算itm_loss
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        device = imgs.device
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        texts = []
        for idx in label:
            texts.append(self.classes_all[idx][1])
        # itc前向过程
        # video_encoder
        video_embeds = self.video_encoder(imgs) # [B, 48, 768]
        video_feat = video_embeds.permute(0,2,1) # [B, 768, 48]
        video_feat = self.video_pool1d(video_feat).squeeze() # [B, 768]
        video_feat = F.normalize(self.vision_proj(video_feat), dim=-1) # [B, 256]
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
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:,0,:]), dim=-1
        ) # [B, 256]
        # 计算真值 sim_targets [B, B] 行和为1
        label = label.view(-1, 1) # [B, 1]
        pos_idx = torch.eq(label, label.t()).float() # [B, B]
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True) # 归一化
        # 计算两种相似度
        sim_i2t = video_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ video_feat.t() / self.temp
        # 计算itc loss
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
        # itm 前向过程，如果一个batch都是同一类，那么不执行判断
        if not self.is_same_class(label):
            # 将首token从cls换成enc
            text_input_ids = text.input_ids.clone()
            text_input_ids[:, 0] = self.tokenizer.enc_token_id
            # 计算出这个batch的正image-text pair
            # 视频正样本仍采用video_embeds
            # 文本正样本text_output_positive通过text_encoder计算得到
            batch_size = label.shape[0]
            # 将itc中计算得到的video_embeds作为text_encoder中的encoder_hidden_states
            video_atts = torch.ones(video_embeds.size()[:-1], dtype = torch.long).to(device) # [B, 48] 全1向量，代表着不需要mask
            text_output_positive = self.text_encoder(
                text_input_ids,
                attention_mask = text.attention_mask,
                encoder_hidden_states = video_embeds,
                encoder_attention_mask = video_atts,
                return_dict = True
            ) # .last_hidden_state [B, 10, 768]
            with torch.no_grad():
                # 只是为了挑出最大负样本，无需计算梯度
                mask = torch.eq(label, label.t()) # [B, B] 正样本对位置为True
                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0) # 把正样本处的权重mask成0，便于找出最大负样本
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)
            # 每一个正文本，找到对应的最大负视频
            video_embeds_negative = []
            for b in range(batch_size):
                negative_idx = torch.multinomial(weights_t2i[b], 1).item()
                video_embeds_negative.append(video_embeds[negative_idx])
            # 每一个正视频，找到对应的最大负文本的id和attn
            text_ids_negative = []
            text_attns_negative = []
            for b in range(batch_size):
                negative_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_negative.append(text_input_ids[negative_idx])
                text_attns_negative.append(text.attention_mask[negative_idx])
            # 构成向量
            video_embeds_negative = torch.stack(
                video_embeds_negative, dim=0
            )
            text_ids_negative = torch.stack(
                text_ids_negative, dim=0
            )
            text_attns_negative = torch.stack(
                text_attns_negative, dim=0
            )
            # [正视频，正文本]经过text_encoder的输出为text_output_positive 已经计算得到
            # 计算[正文本，负视频]和[负文本，正视频]经过text_encoder的输出text_output_negative
            # [正文本，负文本]
            text_ids_all = torch.cat([text_input_ids, text_ids_negative], dim=0)
            text_attns_all = torch.cat([text.attention_mask, text_attns_negative], dim=0)
            # [负视频，正视频]
            video_embeds_all = torch.cat([video_embeds_negative, video_embeds], dim=0)
            video_atts_all = torch.cat([video_atts, video_atts], dim=0)
            # 2B个正负样本对的输出 text_output_negative
            text_output_negative = self.text_encoder(
                text_ids_all, attention_mask = text_attns_all,
                encoder_hidden_states = video_embeds_all,
                encoder_attention_mask = video_atts_all,
                return_dict = True
            ) # .last_hidden_state [2B, 10, 768]
            vl_embeddings = torch.cat(
                [
                    text_output_positive.last_hidden_state[:,0,:],
                    text_output_negative.last_hidden_state[:,0,:]
                ], dim=0
            ) # [B+2B, 768]
            vl_output = self.itm_head(vl_embeddings) # [B+2B, 2]
            vl_labels = torch.cat(
                [
                    torch.ones(batch_size, dtype=torch.long),
                    torch.zeros(2*batch_size, dtype=torch.long)
                ], dim=0
            ).to(device)
            loss_itm = F.cross_entropy(vl_output, vl_labels)
            losses = dict()
            losses['loss_cls'] = (loss_i2t + loss_t2i)/2
            losses['loss_itm'] = loss_itm
            return losses
        else:
            losses = dict()
            losses['loss_cls'] = (loss_i2t + loss_t2i)/2
            losses['loss_itm'] = 0
            return losses    
            

    
    def is_same_class(self, labels):
        return torch.allclose(labels, labels[0])