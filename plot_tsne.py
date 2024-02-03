import numpy as np
from mmcv import load
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

def draw_pic_2(tsne1, tsne2, labels, path = './paper_fig/Fig10.png'):
    # 创建子图布局
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5))
    unque_labs = np.unique(labels)
    taichi_cate = [
        '1.Preparation', '2.Grasp Bird`s Tail', '3.Single Whip', '4.Lift up Hand',
        '5.White Crane Spread its Wings', '6.Brush Knee and Twist Step', '7.Hold the Lute',
        '8.Pulling Blocking and Pounding', '9.Apparent Close Up', '10.Cross Hands'
    ]
    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unque_labs))
    ]
    # 绘制第一个散点图，不标注label
    ax = axes[0]
    for i in range(len(unque_labs)):
        index = np.where(labels == unque_labs[i])
        pi = ax.scatter(
            tsne1[index, 0], tsne1[index, 1], c = [colors[i]]
        )
    ax.set_xticks([])
    ax.set_yticks([])
    # 绘制第二个散点图， 标注label
    ax = axes[1]
    p = []
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labels == unque_labs[i])
        pi = ax.scatter(
            tsne2[index, 0], tsne2[index, 1], c = [colors[i]]
        )
        p.append(pi)
        legends.append(unque_labs[i]+1)
    ax.legend(p, legends, loc = 'upper left', bbox_to_anchor = (-0.2, 0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path)

def draw_pic_1(tsne, labels, path = './paper_fig/Fig10.png'):
    # 创建图形对象
    fig, ax = plt.subplots()
    unque_labs = np.unique(labels)
    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, len(unque_labs))
    ]
    # 绘制散点图， 标注label
    p = []
    legends = []
    for i in range(len(unque_labs)):
        index = np.where(labels == unque_labs[i])
        pi = ax.scatter(
            tsne[index, 0], tsne[index, 1], c = [colors[i]]
        )
        p.append(pi)
        legends.append(unque_labs[i]+1)
    ax.legend(p, legends, loc = 'upper left', bbox_to_anchor = (-0.175, 0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path)

if __name__ == '__main__':
    pca = PCA(n_components=50, random_state=400)
    tsne = TSNE(n_components=2, init = 'pca', n_iter = 3000, random_state = 400)
    pkl_pt = 'model_pth/exp6/t9wswr_wfwc_ft/test_result/t_sne_vis_out.pkl'
    data_label = load(pkl_pt)['target_labels'] # 0到9
    # 数据
    data = load(pkl_pt)['target_outputs']
    pca_data = pca.fit_transform(
        np.array(data)
    )
    tsne_data = tsne.fit_transform(pca_data.real)
    draw_pic_1(
        tsne=tsne_data, 
        labels=data_label,
        path = 'paper_fig/Fig13-4-ft-t9-5p-kp-wswr-wfwc.png'
    )

'''
if __name__ == '__main__':
    pca = PCA(n_components=50, random_state=400)
    tsne = TSNE(n_components=2, init = 'pca', n_iter = 3000, random_state = 400)
    pkl_pt1 = 'model_pth/exp5/t9nsnr_5p_ft_lrd01/test_result/t_sne_vis_out.pkl'
    pkl_pt2 = 'model_pth/exp5/t9nsnr_5p_ft_lrd01_nopre/test_result/t_sne_vis_out.pkl'
    data_label = load(pkl_pt1)['target_labels'] # 0到9
    # 数据1
    data1 = load(pkl_pt1)['target_outputs']
    pca_data1 = pca.fit_transform(
        np.array(data1)
    )
    tsne_data1 = tsne.fit_transform(pca_data1.real)
    # 数据2
    data2 = load(pkl_pt2)['target_outputs']
    pca_data2 = pca.fit_transform(
        np.array(data2)
    )
    tsne_data2 = tsne.fit_transform(pca_data2.real)
    draw_pic_2(
        tsne1=tsne_data1, 
        tsne2=tsne_data2, 
        labels=data_label,
        path = './paper_fig/Fig12.png'
    )
'''