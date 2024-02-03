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

# 使用示例
# 假设你的log文件路径是"log_file.log"
# actionclip日志
# pt1 = 'work_dirs/posec3d/20221025_202224.log'
# poseconv3d日志
pt2 = 'work_dirs/plus/ucf-split1-poseconv3d/20231229_110058.log'
# skeletonclip日志
pt3 = 'work_dirs/plus/ucf-split1-itm/20231230_153454.log'
# skeletonclip++日志
pt4 = 'work_dirs/plus/ucf-split1-pacl/20231226_055535.log'

# loss_ac, acc_ac = extract_data_from_log(pt1)
loss_pc, acc_pc = extract_data_from_log(pt2)
loss_sc, acc_sc = extract_data_from_log(pt3)
loss_scp, acc_scp = extract_data_from_log(pt4)

epoch = len(loss_pc)
batch = len(loss_pc[0])

plt.figure(figsize=(12, 8))
# 绘制loss曲线
for i in range(len(loss_pc)):
    plt.scatter([i+1]*batch, loss_pc[i], color = 'blue', alpha=0.5)
    plt.scatter([i+1]*batch, loss_sc[i], color = 'red', alpha=0.5)
    plt.scatter([i+1]*batch, loss_scp[i], color = 'green', alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 显示散点图的图例在左侧
plt.legend(['Loss PoseConv3D', 'Loss SkeletonClip', 'Loss SkeletonClip++'], loc = 'upper left', bbox_to_anchor=(1, 0.2))
plt.tight_layout()
# 创建第二个y轴绘制识别精度
ax2 = plt.twinx()   
# 绘制识别精度曲线
epochs = np.arange(1, epoch+1)
ax2.plot(epochs, acc_pc*100, 'b-o', label = 'Acc PoseConv3D')
ax2.plot(epochs, acc_sc*100, 'r-s', label = 'Acc SkeletonClip')
ax2.plot(epochs, acc_scp*100, 'g-^', label = 'Acc SkeletonClip++')
# 图表标题和标签

ax2.set_ylabel('Validation Accuracy (%)')
# 图例
ax2.legend(loc = 'upper left', bbox_to_anchor=(1, 1))

plt.savefig('loss_acc.png')

pass

