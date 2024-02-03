model = dict(
    type = 'RecognizerVisecePACL',
    video_encoder = dict(
        type = 'Visece_Video_PACL',
        res_out_channels = 512, # ResNet输出的维度
        in_channels=17,     # 这部分都是ResNet3D_SlowOnly的参数
        base_channels=32,
        num_stages=3,
        out_indices=(2, ),
        stage_blocks=(4, 6, 3),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2)
    ),
    text_encoder = dict(
        type = 'BertModel',
        config = dict(
            architectures = ['BertModel'],
            model_type = 'bert',
            attention_probs_dropout_prob = 0.1,
            hidden_act = 'gelu',
            hidden_dropout_prob = 0.1,
            hidden_size = 768,
            initializer_range = 0.02,
            intermediate_size = 3072,
            layer_norm_eps = 1e-12,
            max_position_embeddings = 512,
            num_attention_heads = 12,
            num_hidden_layers = 12,
            pad_token_id = 0,
            type_vocab_size = 2,
            vocab_size = 30524,
            encoder_width = 768,
            add_cross_attention = True
        ),
        add_pooling_layer = False
    ),
    embed_dim = 256,
    text_encoder_init = 'bert-base-uncased',
    text_lists = 'lists/hmdb51_labels.csv'
)
dataset_type = 'PoseDataset'
ann_file = '/home/yl/public_datasets/heatmap/hmdb51_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(64, 64), keep_ratio=False),
    dict(type='GeneratePoseTarget', with_kp=True, with_limb=False, double=True, left_kp=left_kp, right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu = 16,
    val_videos_per_gpu = 4,
    workers_per_gpu = 4,
    test_dataloader = dict(videos_per_gpu = 4),
    train = dict(
        type = 'RepeatDataset',
        times = 10,
        # times = 1,
        dataset = dict(
            type=dataset_type, 
            ann_file = ann_file, 
            split = 'train2', 
            pipeline = train_pipeline
        )
    ),
    val=dict(
        type = dataset_type,
        ann_file = ann_file,
        split = 'test2',
        pipeline = val_pipeline
    ),
    test = dict(
        type = dataset_type,
        ann_file = ann_file,
        split = 'test2',
        pipeline = test_pipeline
    )
)
# optimizer
optimizer = dict(
    type = 'AdamW', lr = 3.33e-5, weight_decay = 0.01,
    paramwise_cfg = dict(
        custom_keys = {
            'video_encoder': dict(lr_mult = 100, decay_mult = 0.9),
            'text_encoder': dict(lr_mult = 1, decay_mult = 0.9),
            'tokenizer': dict(lr_mult = 1, decay_mult=0.9)
        }
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy = 'CosineAnnealing',
    by_epoch = False,
    min_lr = 0,
    warmup = 'linear',
    warmup_iters = 2,
    warmup_ratio = 0.1,
    warmup_by_epoch = True
)
total_epochs = 24
checkpoint_config = dict(interval=24)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/plus/hmdb-split2-pacl'
find_unused_parameters = True