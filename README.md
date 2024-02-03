## Installation
```shell
git clone https://github.com/eunseo-v/sscls
conda create -n skeletonclip++ python=3.8
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate skeletonclip++
pip install openmim
mim install mmcv-full==1.5.0
mim install mmdet
mim install mmpose
cd skeletonclip++
pip install -r requirements.txt
pip install -e .
```


#### PoseDataset

* 经过PoseDataset后每一个样本的输出只包含results['imgs']: [N, C, T, H, W]和results['label']: tensor([0])

#### Class RecognizerVisece

* 重新写一个Recognizer: 在 `pyskl/models/recognizers/recognizervisece.py` 。将模型注册进工具包中需要包含：
* （1）在recognizervisece.py文件中添加语句 `from ..builder import RECOGNIZERS`
* （2）在类RecognizerVisece前添加语句 `@RECOGNIZERS.register_module()`
* （3）在 `recognizers/__init__.py`中添加 ``from .recognizervisece import RecognizerVisece`` 并且在all中添加'RecognizerVisece'

#### Class RecognizerViseceITM

* 基于RecognizerVisece类，添加正负样本的判断，每一个batch送入的标签，如果都属于同一类，这个batch不去寻找最大负样本进行判断，直接return
* `RecognizerViseceITM`类在文件 `pyskl/models/recognizers/recognizerviseceitm.py`。将模型注册进工具包中需要包含:
* （1）在recognizerviseceitm.py中添加语句 `from ..builder import RECOGNIZERS`。由于需要继承RecognizerVisece类，还需要添加语句 `from .recognizervisece import RecognizerVisece`
* （2）在类RecognizerViseceITM前添加语句 `@RECOGNIZERS.register_module()`
* （3）在 `recognizer/__init__.py`中添加 `from .recognizerviseceitm import RecognizerViseceITM`，并且在all中添加'RecognizerViseceITM'

#### Class RecognizerViseceMo

* 基于RecognizerVisece类，添加动量编码器模块和向量队列，动量编码器的输出进入向量队列，相似度计算使用正常编码器的输出点乘向量队列的内容，标签信息包含动量编码器的相似度和真实的相似度
* `RecognizerVisece`类在文件 `pyskl/models/recognizers/recognizervisecemo.py`。将模型注册进工具包中需要包含：
* （1）在recognizervisecemo.py中添加语句 `from ..builder import RECOGNIZERS`。由于需要需要继承RecognizerVisece类，还需要添加语句 `from .recognizervisece import RecognizerVisece`
* （2）在类RecognizerViseceMo前添加语句 `@RECOGNIZERS.register_module()`
* （3）在 `recognizer/__init__.py`中添加 `from .recognizervisecemo import RecognizerViseceMo`，并且在all中添加'RecognizerViseceMo'

#### Class BertModel

* BertModel相比于blip中的对应的类，初始化config增加了从ConfigDict类到BertConfig类的转换，并且在初始的时候加了一个 `isinstance(config, BertConfig)` 的判断，因为在初始化的时候，如果不添加判断，会报错，将BertModel类注册进工具包需要包含：
* （1）在 `pyskl/models/bert/simplebert.py`文件中添加语句 `from ..builder import BACKBONES`
* （2）在类BertModel前添加语句 `@BACKBONES.register_module()`
* （3）在文件夹 `pyskl/models/bert`中新建文件 `__init__.py`，并且添加语句 `from .simplebert import BertModel`和 `__all__ = ['BertModel']`
* （4）由于新建了文件夹bert，于是需要在 `pyskl/models/__init__.py`中添加 `from .bert import *`

#### Class Visece_Video

* 在ResNet3dSlowOnly的基础上编写Visece_Video类作为video_encoder，类位于 `pyskl/models/cnns/visece_video.py`，将类注册进工具包需要包含：
* （1）在 `pyskl/models/cnns/visece_video.py`文件中添加语句 `from ..builder import BACKBONES`
* （2）在类Visece_Video前添加语句 `@BACKBONES.register_module()`
* （3）在 `pyskl/models/cnns/__init__.py`中添加语句 `from .visece_video import Visece_Video`并在all中添加'Visece_Video'

#### Class Visece_Video_PACL

* 与Class Visece_Video相比去除了self.linear部分，类位于 `pyskl/models/cnns/visece_video.py`，将类注册进工具包需要包含：
* （1）在类Visece_Video_PACL前添加语句 `@BACKBONES.register_module()`
* （2）在 `pyskl/models/cnns/__init__.py`中添加语句 `from .visece_video import Visece_Video_PACL`并在all中添加'Visece_Video_PACL'

#### Class RecognizerVisecePACL

* 重新写一个Recognizer: 在 `pyskl/models/recognizers/recognizervisecepacl.py` 。将模型注册进工具包中需要包含：
* （1）在recognizervisece.py文件中添加语句 `from ..builder import RECOGNIZERS`
* （2）在类RecognizerVisece前添加语句 `@RECOGNIZERS.register_module()`
* （3）在 `recognizers/__init__.py`中添加 ``from .recognizervisecepacl import RecognizerVisecePACL`` 并且在all中添加'RecognizerVisecePACL'
* Class Visece_Video的前向过程中ResNet3dSlowOnly的输出为[B, C=512, T, H, W]，为了得到与文本同一维度[B, C=768]的向量表示，后续操作如下
* （1）reshape [B, CT, H, W]
* （2）AdaptiveAvgPool2d [B, CT, 1, 1]
* （3）reshape [B, T, C=512]
* （4）nn.Linear [B, T, C=768]

#### 训练流程

* 训练集，模型进入 `model.train_step()`，调用 `model.forward()` 进入 `model.forward_train()`完成计算每一个batch的loss的过程
* 验证集，模型提前注册了 `DistEvalHook`，从而由于 `model.forward(return_loss=False)`进入 `model.forward_test()`，计算每一个batch的 `cls_score [B, num_classes]`得分，此处一次验证epoch的每一个batch都要重复计算相同的text_feat，但是我不会把他们写在一次epoch的起始，后续改进

#### 实验结果

##### Class ResNet3dSlowOnly

optimizer: AdamW    lr: 0.01    accuracy: 93.12%  epoch: 24

##### Class RecognizerVisece

| base_lr | video_encoder | text_encoder | tokenizer | accuracy | exp_pth       | epoch |
| ------- | ------------- | ------------ | --------- | -------- | ------------- | ----- |
| 1e-4    | 1e-4          | 1e-4         | 1e-4      | 90.85%   | visece/debug1 | 24    |
| 1e-2    | 1e-2          | 1e-4         | 1e-4      | 91.89%   | visece/debug2 | 24    |
| 1e-3    | 1e-3          | 1e-5         | 1e-5      | 92.32%   | visece/debug3 | 24    |
| 1e-4    | 1e-4          | 1e-5         | 1e-5      | 91.11%   | visece/debug4 | 23    |
| 1e-3    | 1e-3          | 1e-4         | 1e-4      | 89.93%   | visece/debug5 | 9     |
| 1e-2    | 1e-2          | 1e-5         | 1e-5      | 91.73%   | visece/debug6 | 23    |
| 1e-3    | 1e-3          | 0            | 0         | 91.83%   | visece/debug7 | 23    |
| 1e-5    | 1e-3          | 1e-5         | 1e-5      | 92.63%   | visece/debug8 | 21    |
| 1e-4    | 1e-3          | 1e-5         | 1e-5      | 92.48%   | visece/debug9 | 23    |

##### Class RecognizerViseceITM

| base_lr | video_encoder | text_encoder        | tokenizer           | accuracy | exp_pth                | epoch |
| ------- | ------------- | ------------------- | ------------------- | -------- | ---------------------- | ----- |
| 1e-3    | 1e-3          | 1e-5                | 1e-5                | 93.05%   | 43090/visece/itmdebug1 | 23    |
| 1e-5    | 1e-3          | 1e-5                | 1e-5                | 92.79%   | 83090/visece/itmdebug2 | 23    |
| 1e-3    | 1e-3          | 1e-5                | 1e-5                | 92.52%   | 83090/visece/itmdebug1 | 23    |
| 1e-3    | 1e-3          | 0(frozen_bert=True) | 0(frozen_bert=True) | 91.91%   | 43090/visece/itmdebug2 | 24    |

##### Class RecognizerViseceMo

| base_lr | video_encoder | text_encoder | tokenizer | accuracy | exp_pth               | epoch |
| ------- | ------------- | ------------ | --------- | -------- | --------------------- | ----- |
| 1e-3    | 1e-3          | 1e-5         | 1e-5      | 92.36%   | 43090/visece/modebug1 | 24    |
| 1e-3    | 1e-3          | 1e-5         | 1e-5      | 92.32%   | 83090/visece/modebug1 | 20    |
| 1e-3    | 1e-2          | 1e-5         | 1e-5      | 92.66%   | 83090/visece/modebug2 | 23    |
| 1e-5    | 1e-3          | 1e-5         | 1e-5      | 92.37%   | 43090/visece/modebug2 | 24    |
| 1e-5SGD | 1e-3          | 1e-5         | 1e-5      | 89.03%   | 43090/visece/modebug3 | 23    |

#### PACL具体实现代码

核心：计算帧特征与文本特征的相似度一定是使用归一化后的向量

区别：

* 计算后的相似度通过F.sigmoid(activation)或者F.sigmoid(activation*10)或者F.softmax(activation, dim=-1)生成每一帧的权重
* 生成后的权重作用于归一化后的帧特征向量或者作用于embedder的输出，在加权之后再进行归一化

注意事项：

* 归一化后的帧特征向量加权后相加的向量并不是归一化的，需要再次归一化

|                                | F.sigmoid(activation) | F.sigmoid(activation*10)  | F.softmax             |
| ------------------------------ | --------------------- | ------------------------- | --------------------- |
| 作用于归一化后的帧特征向量     |                       |                           | patch_alignment_train |
| 作用于embedder输出的帧特征向量 |                       | patch_alignment_train_mod |                       |

##### def patch_alignment_train()实验思路：对于归一化后的帧特征softmax加权

* （1）（2）（3）步骤一致得到特征为[B, T, C = 512], 在类Visece_Video_PACL 实现
* （4）设计一个embedder $\mathbb{R}^{T \times C=512} \to \mathbb{R}^{T \times C=256}$ ，残差块的形式
* main branch: nn.Linear + ReLU + nn.Linear
* residual connection:  nn.Linear
* （5）计算frame level的相似度  归一化后的特征 ev:[B, T, C=256]  et[B, C=256]需要C的维度上进行点乘相加 ``torch.sum(ev*et.unsqueeze(1), dim=-1)`` 或者使用einsum ``torch.einsum('btc,bc->bt', ev, et)`` s(ev, et):[B, T]
* （6）``a(ev, et) = F.softmax(s(ev, et), dim=-1)`` [B, T]
* （7）计算加权(weighted sum)后的视频特征表示 ``torch.sum(a.unsqueeze(2)*ev, dim=1)`` 或者 ``torch.einsum('bt,btc->bc', a, ev)`` [B, C=256]
* （8）将视频特征和文本特征进行归一化，计算相似度

##### def patch_alignment_val()实验思路：对于归一化后的帧特征softmax加权

* （1）通过video_embeder得到video_feat_raw [B, T, C=256]和L类标签文本的文本特征text_feat_raw [L, C]
* （2）对帧特征和文本特征归一化，然后计算每一个样本的每一帧对于所有L类标签文本的相似度 ``torch.sum((video_feat_raw.unsqueeze(1))*(text_feat_raw.unsqueeze(1).unsqueeze(0)), dim=-1)`` 或者 ``torch.einsum('btc,lc->blt', video_feat_raw, text_feat_raw)``
* （3）``a = F.softmax(s, dim=-1)`` [B, L, T]
* （4）计算每一个样本针对每一类文本的加权视频特征 ``torch.sum( (a.unsqueeze(3))*(video_feat_raw.unsqueeze(1)), dim=2 )`` 或者 ``torch.einsum('blt,btc->blc', a, video_feat_raw)`` [B, L, C]
* （5）将加权的视频特征和之前的文本特征进行归一化，得到video_feat和text_feat
* （6）点积计算相似度 ``torch.sum( video_feat*(text_feat.unsqueeze(0)), dim=-1  )`` 或者 ``torch.einsum('blc,lc->bl', video_feat, text_feat)`` [B, L]
* （7）``F.softmax(cls_score, dim=-1)`` .cpu().numpy()输出
* 新的patch alignment实现
* （1）未进行归一化的video_feat_raw和text_feat_raw送入先进行归一化F.normalize()
* （2）点乘计算相似度，训练过程 `torch.einsum('btc,bc->bt', ev, et)`验证过程 `torch.einsum('btc,lc->blt', video_feat_raw, text_feat_raw)`
* （3）不进行softmax操作，直接数值*10
* （4）使用F.sigmoid()将取值范围压缩到[-1, 1]之间

##### def patch_alignment_train_mod()实验思路：对未归一化的帧特征进行sigmoid加权

* 输入为一个batch内未归一化的视频特征和文本特征
* 两种特征归一化然后点乘计算正样本对之间每一帧的权重
* 权重×10后经过sigmoid函数得到最终的权重
* 加权到未归一化的视频特征后在时间轴上相加得到加权后的视频特征 [B, C]
* 函数输出后经过归一化得到最终的视频特征

##### def patch_alignment_val_mod()实验思路：对未归一化的帧特征进行sigmoid加权

* 输入为一个batch内未归一化的视频特征和**所有类别**的文本特征
* 两种特征归一化然后点乘计算**每一个样本的每一帧对于所有类别文本特征的权重**
* 权重×10后经过sigmoid函数得到最终的权重
* 加权到未归一化的视频特征后在时间轴上相加得到加权后的视频特征(对于每一类文本) [B, L, C]
* 函数输出后经过归一化得到最终的视频特征


##### 模型各路径对应的视频特征计算过程

* pacldebug1

```
采用的是def patch_alignment_train_mod()函数
activations = F.sigmoid(activations * 10)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* pacldebug2

```
采用的是def patch_alignment_train()函数
a_ev_et = F.softmax(s_ev_et, dim=-1)
前向过程没有归一化的阶段
```

* pacldebug3有可能与pacldebug4颠倒

```
采用的是def patch_alignment_train_mod()函数
activations = F.softmax(activations, dim=-1)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* pacldebug4

```
采用的是def patch_alignment_train()函数
a_ev_et = F.softmax(s_ev_et, dim=-1)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

#### 新训练思路

新的训练流程按照原有的验证流程，对所有文本都计算alignment后视频特征，只有正样本相似度为1，其余相似度均为0，以避免过拟合问题

##### 实验设置训练集重复次数为1

* paclmod1: 89.62%

```
采用的是def patch_alignment_val()函数
a_ev_et = F.softmax(s_ev_et, dim=-1)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* paclmod2: 88.82%

```
采用的是def patch_alignment_val()函数
activations = F.sigmoid(activations*10)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* paclmod3: 89.73%

```
采用的是def patch_alignment_val_mod()函数
activations = F.softmax(activations, dim=-1)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* paclmod4: 88.84%

```
采用的是def patch_alignment_val_mod()函数
activations = F.sigmoid(activation*10)
前向过程video_feat = F.normalize(video_feat, dim=-1)
```

* paclmod5: 90.06%   92.64%

```
采用的是def patch_alignment_val_mod()函数
activations = F.softmax(activations, dim=-1)
前向过程video_feat = F.normalize(video_feat, dim=-1)
embedder采用visece的embedder
```

* visecemod1采用与pacl相同的video_embedder : 88.22%

```
class Visece_Video的前向过程中删除最后一句 x=self.linear(x),此时输出维度为512
原vision_projector作用于时间轴pool1d()后的视频特征向量
修改后的的video_embedder作用于未进行时间轴pool1d()的帧特征向量。
```

* visecemod2采用与pacl相同的video_embedder，作用位置不同: 88.06%

```
class Visece_Video的前向过程中删除最后一句 x = self.linear(x)，此时输出维度为512
修改后的video_embedder作用于时间轴pool1d()后的视频特征向量
```

* visecemod3采用linear的embedder: 88.39%

```
class Visece_Video的前向过程中删除最后一句 x=self.linear(x),此时输出维度为512
vision_projector作用于时间轴pool1d()后的视频特征向量
```

* visecemod4采用linear的embedder，作用位置不同：88.21%

```
class Visece_Video的前向过程中删除最后一句 x=self.linear(x),此时输出维度为512
vision_projector作用于未进行时间轴pool1d()的帧特征向量
```
