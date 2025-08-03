import numpy as np
import os

def fix_sdf_dimensions():
    """修复SDF数据的维度顺序以匹配PNG序列"""
    
    sdf_path = 'sdf_grid.npy'
    sdf_path_fixed = 'sdf_grid_fixed.npy'
    
    print("正在加载原始SDF数据...")
    sdf_data = np.load(sdf_path)
    print(f"原始SDF形状: {sdf_data.shape}")
    
    # 重新排列维度顺序：从 (796, 795, 552) 到 (552, 795, 796)
    # 这相当于从 (H, W, D) 转换为 (D, W, H)
    sdf_fixed = np.transpose(sdf_data, (2, 1, 0))
    print(f"修复后SDF形状: {sdf_fixed.shape}")
    
    # 保存修复后的数据
    np.save(sdf_path_fixed, sdf_fixed)
    print(f"修复后的SDF数据已保存到: {sdf_path_fixed}")
    
    # 验证新数据
    print("\n验证修复后的数据:")
    print(f"数据类型: {sdf_fixed.dtype}")
    print(f"值范围: [{sdf_fixed.min():.4f}, {sdf_fixed.max():.4f}]")
    print(f"均值: {np.mean(sdf_fixed):.4f}")
    print(f"标准差: {np.std(sdf_fixed):.4f}")
    
    return sdf_path_fixed

if __name__ == '__main__':
    fixed_path = fix_sdf_dimensions()
    print(f"\n✅ SDF维度修复完成！请使用文件: {fixed_path}")
