import os
import yaml
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import traceback
import argparse
import time

import datasets
import models
import utils
from datasets.queue import dequeue_and_enqueue
from datasets.degrade import SRMDPreprocessing

def create_config_for_real_data(config_path, liver_data_path, sdf_data_path, test_params):
    """创建基于真实数据的配置文件"""
    
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))

    # 检查数据文件是否存在
    if not os.path.exists(liver_data_path):
        raise FileNotFoundError(f"Liver数据目录不存在: {liver_data_path}")
    
    if not os.path.exists(sdf_data_path):
        raise FileNotFoundError(f"SDF数据文件不存在: {sdf_data_path}")
    
    # 计算SDF统计信息
    print(f"正在加载SDF数据: {sdf_data_path}")
    sdf_data = np.load(sdf_data_path).astype(np.float64)
    sdf_mean = np.mean(sdf_data)
    sdf_std = np.std(sdf_data)
    
    print(f"SDF数据形状: {sdf_data.shape}")
    print(f"SDF均值: {sdf_mean}")
    print(f"SDF标准差: {sdf_std}")

    # 获取绝对路径
    liver_data_abs_path = os.path.abspath(liver_data_path)
    sdf_data_abs_path = os.path.abspath(sdf_data_path)

    # 创建配置文件内容
    config_content = f"""
seed: 42
lambda_geom: {test_params['lambda_geom']}
batch_size: {test_params['batch_size']}
total_batch_size: {test_params['batch_size']}
sample_q: {test_params['sample_q']}
inp_size: {test_params['inp_size']}
queue_size: {test_params['batch_size'] * 2}

train_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {liver_data_abs_path}, sdf_path: {sdf_data_abs_path}, repeat: 1}}}}
  wrapper: {{name: sr-gaussian, args: {{inp_size: {test_params['inp_size']}, sample_q: {test_params['sample_q']}, augment: false, scale: {test_params['scale']}}}}}

val_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {liver_data_abs_path}, sdf_path: {sdf_data_abs_path}}}}}
  wrapper: {{name: sr-gaussian, args: {{inp_size: {test_params['inp_size']}, sample_q: {test_params['sample_q']}, scale: {test_params['scale']}}}}}

data_norm: {{inp: {{sub: [0.5], div: [0.5]}}, gt: {{sub: [0.5], div: [0.5]}}}}
sdf_norm: {{sub: [{sdf_mean}], div: [{sdf_std if sdf_std > 1e-6 else 1.0}]}}
optimizer: {{name: adam, args: {{lr: 1.e-4}}}}
epoch_max: 1

model:
  name: models
  args: {{}}
  SR:
    name: liif
    args:
      encoder_spec: {{name: rdn, args: {{no_upsampling: true}}}}
      sdf_head: true
      imnet_spec: {{name: mlp, args: {{out_dim: 1, hidden_list: [32, 32]}}}}
  degrade:
    name: simsiam
    args: {{dim: 32}}
  path: null
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"配置文件已创建: {config_path}")

def run_training_test(config_path, test_params, num_steps=1):
    """运行训练测试"""
    
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    train_loader, _ = datasets.make_data_loaders(config, DDP=False, state='SR')
    print("数据加载器创建成功。")

    model = models.make(config['model'], args={'config': config}).to(device)
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    print("模型和优化器创建成功。")
    
    sr_criterion = nn.L1Loss()
    geom_criterion = nn.MSELoss()
    degrade_op = SRMDPreprocessing()
    pool = dequeue_and_enqueue(config, 'SR').to(device)
    
    print(f"\n开始训练测试，运行 {num_steps} 个步骤...")
    
    for step in range(num_steps):
        print(f"\n=== 步骤 {step + 1}/{num_steps} ===")
        start_time = time.time()
        
        batch = next(iter(train_loader))
        print("成功获取一个批次的数据。")
        
        for k, v in batch.items():
            if v is not None:
                batch[k] = v.to(device)
                
        lr = degrade_op(batch['inp'], scale=test_params['scale'], norm=True)
        p = {'lr': lr, 'gt': batch['gt'], 'cell': batch['cell'], 'coord': batch['coord'], 
             'scale': batch['scale'].type(torch.FloatTensor), 'gt_sdf': batch.get('gt_sdf')}
        lr, gt, cell, coord, scale, gt_sdf = pool(p)
        
        # 前向传播
        optimizer.zero_grad()
        model.train()
        pred_rgb = model(lr, coord, cell, state='train')
        pred_sdf = model.SR.sdf_pred
        
        # 计算损失
        loss_sr = sr_criterion(pred_rgb, gt)
        total_loss = loss_sr
        loss_geom_val = 0.0
        
        if pred_sdf is not None and gt_sdf is not None:
            # 方案A: 直接在归一化空间计算SDF损失，避免反归一化的数值问题
            # 这样可以避免大数值导致的梯度爆炸问题
            loss_geom = geom_criterion(pred_sdf, gt_sdf)
            total_loss = total_loss + config['lambda_geom'] * loss_geom
            loss_geom_val = loss_geom.item()
            
            # 如果需要监控原始空间的损失，可以添加以下代码（仅用于打印，不参与梯度计算）
            # with torch.no_grad():
            #     t = config['sdf_norm']
            #     sdf_sub = t['sub'][0]
            #     sdf_div = t['div'][0]
            #     pred_sdf_orig = pred_sdf * sdf_div + sdf_sub
            #     gt_sdf_orig = gt_sdf * sdf_div + sdf_sub
            #     loss_geom_orig = geom_criterion(pred_sdf_orig, gt_sdf_orig)
            #     print(f"    - 原始空间SDF损失: {loss_geom_orig.item():.4f}")

        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        step_time = time.time() - start_time
        print(f"  - SR Loss: {loss_sr.item():.4f}")
        print(f"  - Geom Loss: {loss_geom_val:.4f}")
        print(f"  - Total Loss: {total_loss.item():.4f}")
        print(f"  - 步骤用时: {step_time:.2f}秒")

def validate_data_consistency(liver_data_path, sdf_data_path):
    """验证数据一致性"""
    print("\n=== 数据一致性验证 ===")
    
    # 检查liver_data中的图片数量
    liver_files = [f for f in os.listdir(liver_data_path) if f.endswith('.png')]
    print(f"Liver数据文件数量: {len(liver_files)}")
    
    # 检查SDF数据
    sdf_data = np.load(sdf_data_path)
    print(f"SDF数据形状: {sdf_data.shape}")
    print(f"SDF数据类型: {sdf_data.dtype}")
    print(f"SDF值范围: [{sdf_data.min():.4f}, {sdf_data.max():.4f}]")
    
    # 检查一张示例图片
    if liver_files:
        sample_img_path = os.path.join(liver_data_path, liver_files[0])
        sample_img = io.imread(sample_img_path)
        print(f"示例图片形状: {sample_img.shape}")
        print(f"示例图片数据类型: {sample_img.dtype}")
        print(f"示例图片值范围: [{sample_img.min()}, {sample_img.max()}]")
    
    # 验证形状匹配
    expected_shape = (len(liver_files), sample_img.shape[0], sample_img.shape[1])
    if sdf_data.shape == expected_shape:
        print(f"✅ 数据形状匹配: SDF {sdf_data.shape} == 期望 {expected_shape}")
    else:
        print(f"❌ 数据形状不匹配: SDF {sdf_data.shape} != 期望 {expected_shape}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="在真实liver数据上测试训练流水线")
    parser.add_argument('--liver_data', default='liver_data', help='Liver数据目录路径')
    parser.add_argument('--sdf_data', default='sdf_grid_fixed.npy', help='SDF数据文件路径')
    parser.add_argument('--config', default='configs/test_real_training.yaml', help='输出配置文件路径')
    parser.add_argument('--steps', type=int, default=3, help='运行的训练步骤数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--inp_size', type=int, default=48, help='输入尺寸')
    parser.add_argument('--sample_q', type=int, default=2304, help='采样点数量')
    parser.add_argument('--scale', type=int, default=8, help='缩放因子')
    parser.add_argument('--lambda_geom', type=float, default=0.1, help='几何损失权重')
    parser.add_argument('--keep_config', action='store_true', help='保留生成的配置文件')
    
    args = parser.parse_args()
    
    test_params = {
        'batch_size': args.batch_size,
        'inp_size': args.inp_size,
        'sample_q': args.sample_q,
        'scale': args.scale,
        'lambda_geom': args.lambda_geom
    }
    
    try:
        print("🚀 开始真实数据训练流水线测试")
        print(f"参数配置: {test_params}")
        
        # 验证数据一致性
        if not validate_data_consistency(args.liver_data, args.sdf_data):
            print("❌ 数据验证失败")
            return
        
        # 创建配置文件
        print(f"\n=== 创建配置文件 ===")
        create_config_for_real_data(args.config, args.liver_data, args.sdf_data, test_params)
        
        # 运行训练测试
        print(f"\n=== 开始训练测试 ===")
        run_training_test(args.config, test_params, args.steps)
        
        print(f"\n✅ 测试成功完成！运行了 {args.steps} 个训练步骤。")
        
    except Exception as e:
        print(f"\n❌ 测试失败，出现错误: {e}")
        traceback.print_exc()
    finally:
        # 清理配置文件（除非用户选择保留）
        if not args.keep_config and os.path.exists(args.config):
            try:
                os.remove(args.config)
                print(f"已删除临时配置文件: {args.config}")
            except OSError as e:
                print(f"删除配置文件时出错: {e}")

if __name__ == '__main__':
    main()
