import torch
#import keras
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt

def save_slice_as_image(slice, filename):
    """
    保存数组切片为图片。
    """
    plt.figure()
    plt.imshow(slice, cmap='gray')  # 使用灰度颜色映射
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_center_slice_images(data, file_name, output_path):
    """
    Save the center slices of the data as images.
    Args:
        data (numpy.ndarray): The 3D data array.
        file_name (str): The original data file name.
        output_path (str): The directory path to save images.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    center_i = data.shape[0] // 2
    center_j = data.shape[1] // 2
    center_k = data.shape[2] // 2

    plt.figure(figsize=(10, 10))
    plt.imshow(data[center_i, :, :], cmap='gray')
    plt.colorbar()
    plt.title('Center Slice (i-axis)')
    image_path_i = os.path.join(output_path, file_name.replace('.dat', '_center_slice_i.png'))
    plt.savefig(image_path_i)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(data[:, center_j, :], cmap='gray')
    plt.colorbar()
    plt.title('Center Slice (j-axis)')
    image_path_j = os.path.join(output_path, file_name.replace('.dat', '_center_slice_j.png'))
    plt.savefig(image_path_j)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(data[:, :, center_k], cmap='gray')
    plt.colorbar()
    plt.title('Center Slice (k-axis)')
    image_path_k = os.path.join(output_path, file_name.replace('.dat', '_center_slice_k.png'))
    plt.savefig(image_path_k)
    plt.close()

    print(f"Saved center slice images: {image_path_i}, {image_path_j}, {image_path_k}")

def addnoise(gx):
    noise_std = 0.01  # 设置噪声的标准差
    noise = np.random.normal(0, noise_std, gx.shape)  # 生成噪声
    gx += noise  # 将噪声添加到地震数据
    return gx

def normalize2(sx):
    sxmean = np.mean(sx)
    sxstd = np.std(sx)
    sx = (sx-sxmean)/sxstd
    return sx

def normalize(gx):
    vmin_s, vmax_s = -3, 3
    # 线性拉伸到0-255范围内
    vmin, vmax = np.min(gx), np.max(gx)
    gx = (gx - vmin) * 255.0 / (vmax - vmin)
    # 计算直方图
    hist, bins = np.histogram(gx, bins=256, range=(0, 255))
    # 计算直方图均衡化
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    gx_eq = np.interp(gx.flatten(), bins[:-1], cdf).reshape(gx.shape)
    # 将灰度图像值域映射回数据值域范围
    gx_eq = gx_eq / 255.0 * (vmax_s - vmin_s) + vmin_s
    gx_eq.astype(np.single)
    return gx_eq

class MyDataset(Dataset):
    def __init__(self, dpth, fpth ,data_IDs, dimension, chann):
        self.dpth = dpth
        self.fpth = fpth
        self.dim = dimension
        self.chann = chann
        self.data_IDs = data_IDs
        
        # 筛选指定的文件
        self.dfile = [os.path.join(self.dpth, file) for file in self.data_IDs]
        self.ffile = [os.path.join(self.fpth, file) for file in self.data_IDs]
        
        self.num_file = len(self.data_IDs)  # 由data_IDs的长度决定
        self.new_dim = (216, 216, 216)  # 这里设置新的子体积维度

    def __getitem__(self, item):
        a = 1 #data augumentation
        
        # 计算提取128x128x128子体积的起始和结束索引
        new_dim = (216, 216, 216)

        gx = np.fromfile(self.dfile[item], dtype=np.single)
        fx = np.fromfile(self.ffile[item], dtype=np.uint8)
        fx = fx.astype(np.single)
        # 打印原始数据的维度
        # print(f"Original gx shape: {gx.shape}")
        # print(f"Original fx shape: {fx.shape}")

        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)
        gx = np.transpose(gx)
        fx = np.transpose(fx)
        
        # 打印重塑后的数据维度，以验证是否与期望一致
        # print(f"Reshaped gx shape: {gx.shape}")
        # print(f"Reshaped fx shape: {fx.shape}")

        # 计算中心点坐标
        center = [d // 2 for d in self.dim]  # (112, 112, 112)
        start = [c - n // 2 for c, n in zip(center, new_dim)]  # 计算起始点
        end = [s + n for s, n in zip(start, new_dim)]  # 计算结束点
        
        # 提取子体积
        gx_sub = gx[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        fx_sub = fx[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        # print(f"Reshaped gx shape: {gx_sub.shape}")
        # print(f"Reshaped fx shape: {fx_sub.shape}")
        gx_sub = normalize2(gx_sub)

        # center_slice_index = gx.shape[1] // 4
        # gx_center_slice = gx_sub[:, center_slice_index, :]
        # fx_center_slice = fx_sub[:, center_slice_index, :]

        # save_slice_as_image(gx_center_slice, f'gx_normal_{item}.png')
        # save_slice_as_image(fx_center_slice, f'fx_normal_{item}.png')

        # gx2 = np.rot90(gx_sub, 1, (1,2))
        # fx2 = np.rot90(fx_sub, 1, (1,2))
        # gx_center_slice2 = gx2[:, center_slice_index, :]
        # fx_center_slice2 = fx2[:, center_slice_index, :]

        # save_slice_as_image(gx_center_slice2, f'gx_rot_{item}.png')
        # save_slice_as_image(fx_center_slice2, f'fx_rot_{item}.png')

        # exit(0)

        # augment
        X = np.zeros((a, 1, *new_dim), dtype=np.single)
        for i in range(a):
            X[i, :] = np.reshape(np.rot90(gx_sub, i, (1, 2)), (1,*new_dim))

        Y = np.zeros((a, *new_dim), dtype=np.single)
        for i in range(a):
            Y[i, :] = np.reshape(np.rot90(fx_sub, i, (1, 2)), (new_dim))

        # X = gx
        # Y = fx
        # X = X[np.newaxis, :]
        return X, Y

    def __len__(self):
        return self.num_file


if __name__ == '__main__':
    dim = (256, 256, 256)
    chann = 1
    datas = MyDataset('G:/datas/Train/seis/','G:/datas/Train/channel/',dim,chann)
    (a,b) = datas[4]
    print(a.shape,b.shape)
    print(b.shape)
    # print(len(datas))

