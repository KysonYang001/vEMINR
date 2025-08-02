import argparse
from skimage import io
import numpy as np

def main():
    """
    一个简单的工具，用于计算3D TIFF体数据（volume data）的全局均值和标准差。
    """
    parser = argparse.ArgumentParser(description="为3D TIFF体数据计算均值和标准差。")
    parser.add_argument('--path', type=str, required=True, help='SDF .tif 体数据文件的路径。')
    args = parser.parse_args()

    print(f"正在从以下路径加载体数据: {args.path}")
    try:
        volume = io.imread(args.path)
    except FileNotFoundError:
        print(f"错误: 在路径 {args.path} 未找到文件。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    print(f"体数据已加载，形状: {volume.shape}，数据类型: {volume.dtype}")

    # 为避免计算中出现溢出或下溢问题，确保数据为浮点类型
    volume = volume.astype(np.float64)

    # 计算均值和标准差
    print("正在计算均值和标准差...")
    mean = np.mean(volume)
    std = np.std(volume)

    print("\n--- 统计结果 ---")
    print(f"均值 (Mean): {mean}")
    print(f"标准差 (Standard Deviation): {std}")
    print("------------------\n")
    print("请将这些值添加到您的训练配置文件中的 'sdf_norm' 部分。")
    print("示例:")
    print("sdf_norm:")
    print(f"  sub: [{mean}]")
    print(f"  div: [{std}]")

if __name__ == '__main__':
    main()