import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import torch
# 生成示例数据（1000个epoch）
# epochs = np.arange(1, 1001)
# psnr_values = 30 + 10 * (1 - np.exp(-steps/200)) + np.random.normal(0, 0.2, 1000)


from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def norm01(image):
    return (image - image.min())/(image.max()-image.min())

def norm(image):
    return image * 0.5 + 0.5

def show_imgs(images, nrow=1, title=''):
    #
    # images = norm(images)

    n_img = images.shape[0]
    n_col = min(n_img, nrow)
    n_row = int(np.ceil(n_img / n_col))

    grid_img = make_grid(images, nrow=n_row, padding=2)

    # 可视化网格图片
    plt.figure(figsize=(10, 5))
    plt.imshow(grid_img.permute(1, 2, 0))  # 调整通道顺序以适应 matplotlib 的要求
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_imgs_np(images, nrow=2, title=''):
    n_img = len(images)
    n_col = min(n_img, nrow)
    n_row = int(np.ceil(n_img / n_col))

    # 创建一个新的图形
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))

    # 遍历每个图像并显示在对应的子图中
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 关闭坐标轴

    # 设置图形标题并显示
    plt.suptitle(title)
    plt.show()


def visualize_orthogonal_views(volume):
    """
    可视化3D体数据的三个正交切面 (XY, XZ, YZ) 并保存为图像。

    Args:
        volume (np.ndarray): 输入的3D体数据，期望的维度顺序是 (Z, Y, X)。
        save_path (str): 保存可视化结果的图像文件路径 (例如 'output/views.png')。
    """
    # 确保输入是三维的
    if volume.ndim != 3:
        raise ValueError(f"输入体数据需要是3维, 但接收到 {volume.ndim} 维")

    # 获取体数据的三个维度大小 (假设顺序为 Z, Y, X)
    shape_z, shape_y, shape_x = volume.shape
    print(f"开始可视化，体数据形状 (Z, Y, X): ({shape_z}, {shape_y}, {shape_x})")

    # 计算每个维度的中心切片索引
    mid_z, mid_y, mid_x = shape_z // 2, shape_y // 2, shape_x // 2

    # 提取三个正交切面
    slice_xy = volume[mid_z, :, :]  # XY 平面 (在Z轴的中间)
    slice_xz = volume[:, mid_y, :]  # XZ 平面 (在Y轴的中间)
    slice_yz = volume[:, :, mid_x]  # YZ 平面 (在X轴的中间)

    # 创建一个 1x3 的子图来并排显示三个切面
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Orthogonal Views of the Reconstructed Volume', fontsize=16)

    # 显示 XY 切面
    axes[0].imshow(slice_xy, cmap='gray', origin='lower')
    axes[0].set_title(f'XY Plane (at Z={mid_z})')
    axes[0].set_xlabel('X-axis')
    axes[0].set_ylabel('Y-axis')

    # 显示 XZ 切面
    axes[1].imshow(slice_xz, cmap='gray', origin='lower')
    axes[1].set_title(f'XZ Plane (at Y={mid_y})')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Z-axis')

    # 显示 YZ 切面
    axes[2].imshow(slice_yz, cmap='gray', origin='lower')
    axes[2].set_title(f'YZ Plane (at X={mid_x})')
    axes[2].set_xlabel('Z-axis')
    axes[2].set_ylabel('Y-axis')

    # 调整布局并保存图像
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # 如果希望在运行时直接看到图像，可以取消这行注释


import math
def plot_line(steps, values, title='PSNR Progression with Color Mapping', ylabel='value'):

    length = len(steps)

    plt.figure(figsize=(14, 6))

    # 将线段转换为可映射颜色的集合
    points = np.array([steps, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建颜色映射
    norm = Normalize(values.min(), values.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1)
    lc.set_array(values)  # 绑定PSNR值到颜色

    # 绘制并添加colorbar
    plt.gca().add_collection(lc)
    cbar = plt.colorbar(lc, ax=plt.gca(), label=ylabel)

    # 标注关键点（保留最大值/最小值标注）
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)
    plt.scatter(steps[max_idx], values[max_idx], c='red', s=60, zorder=5,
                label=f'Max: {values[max_idx]:.4f}dB')
    plt.scatter(steps[min_idx], values[min_idx], c='blue', s=60, zorder=5,
                label=f'Min: {values[min_idx]:.4f}dB')

    # 坐标轴设置

    res = values.max() - values.min()
    res = res * 0.2
    plt.xlim(steps.min(), steps.max())
    plt.ylim(values.min() - res, values.max() + res)
    # plt.xticks(np.arange(0, 1001, 100))
    plt.title(title, fontsize=14)
    plt.xlabel('step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

def plot_1d_distribution(data_tensor, data_min=0, data_max=1, label=''):
    """
    一维数据点分布可视化

    参数：
    data_tensor : torch.Tensor - 输入的一维数据张量
    data_min    : float       - 坐标系左端点
    data_max    : float       - 坐标系右端点
    """
    # 转换为numpy数组并展平

    if data_tensor.device != torch.device('cpu'):
        data_tensor = data_tensor.to(torch.device('cpu'))
    data = data_tensor.numpy().flatten()

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 2))

    # 绘制基线（水平轴）
    ax.hlines(0, data_min, data_max, colors='black', linewidth=1)

    # 绘制端点标记
    ax.vlines([data_min, data_max], -0.1, 0.1, colors='red', linewidth=2)
    ax.text(data_min, -0.2, f"Min: {data_min:.2f}", ha='center')
    ax.text(data_max, -0.2, f"Max: {data_max:.2f}", ha='center')

    # 绘制数据点（带轻微垂直抖动）
    # y_jitter = np.random.uniform(-0.05, 0.05, size=len(data))
    y_jitter = np.random.uniform(0, 0, size=len(data))
    ax.scatter(data, y_jitter, alpha=0.7, c='blue', edgecolors='white')

    # 坐标轴设置
    ax.set_xlim(data_min - 0.1 * (data_max - data_min), data_max + 0.1 * (data_max - data_min))
    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])  # 隐藏y轴
    ax.set_title(f"time step Distribution, {label}", pad=20)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_tensor(tensor: torch.Tensor,
                title: str = "1D Tensor Visualization",
                save_path: str = None):
    """
    绘制一维张量的索引-数值图

    参数：
    tensor    : 输入的一维 PyTorch 张量
    title     : 图表标题（可选）
    save_path : 图片保存路径（可选）
    """
    # 转换为 NumPy 数组并确保在 CPU
    # data = tensor.detach().cpu().numpy().flatten()
    data = tensor.flatten()
    # 生成索引
    indices = range(len(data))

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制折线图
    plt.plot(indices, data,
             marker='o',  # 数据点标记为圆圈
             linestyle='-',  # 实线连接
             linewidth=1,
             markersize=4,
             color='#2E86C1',  # 设定颜色
             alpha=0.8)  # 透明度

    # 美化图表
    plt.title(title, fontsize=14)
    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 自动调整坐标轴范围
    plt.xlim(-0.5, len(data) + 0.5)
    plt.ylim(min(data) - 0.1, max(data) + 0.1)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    else:
        plt.show()

def plot_psnr_ssim(steps, psnr, ssim):
    # steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # psnr = [30, 32, 33, 35, 36, 36.5, 37, 37.5, 38, 38.2]
    # ssim = [0.85, 0.86, 0.87, 0.88, 0.89, 0.895, 0.90, 0.91, 0.92, 0.93]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 使用蓝色绘制PSNR，并添加标注
    color_psnr = 'tab:blue'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('PSNR', color=color_psnr)
    ln1, = ax1.plot(steps, psnr, color=color_psnr, marker='o', label='PSNR')
    ax1.tick_params(axis='y', labelcolor=color_psnr)

    ax1.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)

    # 创建共享x轴的第二个y轴，使用红色绘制SSIM，并添加标注
    ax2 = ax1.twinx()
    color_ssim = 'tab:red'
    ax2.set_ylabel('SSIM', color=color_ssim)
    ln2, = ax2.plot(steps, ssim, color=color_ssim, marker='s', label='SSIM')
    ax2.tick_params(axis='y', labelcolor=color_ssim)

    # 合并图例
    lines = [ln1, ln2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title('PSNR and SSIM')
    plt.tight_layout()
    plt.show()


def plot_lpips(steps, lpips):
    # steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # psnr = [30, 32, 33, 35, 36, 36.5, 37, 37.5, 38, 38.2]

    plt.figure(figsize=(10, 6))

    # 绘制 PSNR
    plt.plot(steps, lpips, color='tab:blue', marker='o', label='PSNR')

    # 添加网格线
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    # 添加标签和标题
    plt.xlabel('Steps')
    plt.ylabel('PSNR')
    plt.title('PSNR')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # 示例数据

    # 示例数据
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    psnr = [30, 32, 33, 35, 36, 36.5, 37, 37.5, 38, 38.2]

    plt.figure(figsize=(10, 6))

    # 绘制 PSNR
    plt.plot(steps, psnr, color='tab:blue', marker='o', label='PSNR')

    # 添加网格线
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    # 添加标签和标题
    plt.xlabel('Steps')
    plt.ylabel('PSNR')
    plt.title('PSNR随Steps的变化趋势')
    plt.legend()

    plt.show()