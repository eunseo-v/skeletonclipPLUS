import re
import matplotlib.pyplot as plt
import numpy as np

def extract_data_from_log(file_path):
    # 正则表达式，用于匹配指定的行
    pattern = r"Epoch \[(\d+)\].*loss: ([\d\.]+),"
    pattern_val = r"Epoch\(val\) \[(\d+)\].*top1_acc: ([\d\.]+),"

    # 存储提取的数据
    extracted_data = []
    val_acc = []

    # 读取文件并逐行处理
    with open(file_path, "r") as file:
        for line in file:
            # 使用正则表达式匹配行
            match = re.search(pattern, line)
            if match:
                # 提取Epoch值和loss值
                epoch = int(match.group(1))
                loss = float(match.group(2))
                if len(extracted_data)<epoch:
                    extracted_data.append([loss])
                else:
                    extracted_data[-1].append(loss)
            match_val = re.search(pattern_val, line)
            if match_val:
                # 提取Epoch值和top1_acc值
                top1_acc = float(match_val.group(2))
                val_acc.append(top1_acc)

    return np.float32(extracted_data), np.float32(val_acc)

# ucf101-split1实验结果
# poseconv3d日志
pt1 = 'work_dirs/plus/ucf-split1-poseconv3d/20231229_110058.log'
# skeletonclip日志
pt2 = 'work_dirs/plus/ucf-split1-itm/20231230_153454.log'
# skeletonclip++日志
pt3 = 'work_dirs/plus/ucf-split1-pacl/20231226_055535.log'

# hmdb51-split1
# poseconv3d日志
# pt1 = 'work_dirs/plus/hmdb-split1-poseconv3d/20231229_005519.log'
# skeletonclip日志
# pt2 = 'work_dirs/plus/hmdb-split1-itm/20231227_044708.log'
# skeletonclip++日志
# pt3 = 'work_dirs/plus/hmdb-split1-pacl/20231226_055334.log'

loss_pc, acc_pc = extract_data_from_log(pt1)
loss_sc, acc_sc = extract_data_from_log(pt2)
loss_scp, acc_scp = extract_data_from_log(pt3)

epochs = len(loss_pc) 
batches_per_epoch = len(loss_pc[0])
batch_numbers = np.arange(1, epochs*batches_per_epoch+1)

# 创建图表
plt.figure(figsize=(12, 8))
# 绘制loss曲线，连接所有batch的loss
plt.plot(batch_numbers, loss_pc.reshape(-1), color = 'blue', label = 'Loss PoseConv3D')
plt.plot(batch_numbers, loss_sc.reshape(-1), color = 'red', label = 'Loss SkeletonCLIP')
plt.plot(batch_numbers, loss_scp.reshape(-1), color = 'green', label = 'Loss SkeletonCLIP++')
plt.legend(loc = 'upper left', bbox_to_anchor = (1, 0.2))
plt.xlabel('Epoch', fontsize = 15)
plt.ylabel('Loss',fontsize = 15)
plt.xticks(
    ticks=np.arange(batches_per_epoch, epochs*batches_per_epoch+1, batches_per_epoch),
    labels = np.arange(1, epochs+1), fontsize = 10
)
ax2 = plt.twinx()  
# 在每个epoch结束时绘制验证精度标记
ax2.plot(
    np.arange(batches_per_epoch, (epochs+1)*batches_per_epoch, batches_per_epoch), 
    acc_pc*100, 'b-o', label = 'Acc PoseConv3D'
)
ax2.plot(
    np.arange(batches_per_epoch, (epochs+1)*batches_per_epoch, batches_per_epoch), 
    acc_sc*100, 'r-s', label = 'Acc SkeletonCLIP'
)
ax2.plot(
    np.arange(batches_per_epoch, (epochs+1)*batches_per_epoch, batches_per_epoch), 
    acc_scp*100, 'g-^', label = 'Acc SkeletonCLIP++'
)

ax2.set_ylabel('Validation Accuracy (%)',fontsize = 15)
ax2.legend(loc = 'upper left', bbox_to_anchor=(1, 0.95))
plt.tight_layout()
plt.savefig('loss_acc_ucf101.png')