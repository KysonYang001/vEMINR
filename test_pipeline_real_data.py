import os
import yaml
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import shutil
import traceback

import datasets
import models
import utils
from datasets.queue import dequeue_and_enqueue
from datasets.degrade import SRMDPreprocessing

# --- 1. 定义测试参数和真实数据路径 ---
REAL_CONFIG_PATH = 'configs/test_pipeline_real.yaml'
LIVER_DATA_PATH = 'liver_data'
SDF_DATA_PATH = 'sdf_grid_fixed.npy'  # 使用修复后的SDF文件
CONFIG_DIR = 'configs'

TEST_PARAMS = {
    'batch_size': 2,
    'inp_size': 48, 
    'sample_q': 2304,
    'scale': 8,
    'lambda_geom': 0.1
}

# --- 2. 创建基于真实数据的配置文件 ---
def create_real_data_config():
    """生成基于真实liver数据的配置文件"""
    print("--- 步骤1: 创建基于真实数据的配置文件 ---")
    
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    # 检查数据文件是否存在
    if not os.path.exists(LIVER_DATA_PATH):
        raise FileNotFoundError(f"Liver数据目录不存在: {LIVER_DATA_PATH}")
    
    if not os.path.exists(SDF_DATA_PATH):
        raise FileNotFoundError(f"SDF数据文件不存在: {SDF_DATA_PATH}")
    
    # 计算SDF统计信息
    print(f"正在加载SDF数据: {SDF_DATA_PATH}")
    sdf_data = np.load(SDF_DATA_PATH).astype(np.float64)
    sdf_mean = np.mean(sdf_data)
    sdf_std = np.std(sdf_data)
    
    print(f"SDF数据形状: {sdf_data.shape}")
    print(f"SDF均值: {sdf_mean}")
    print(f"SDF标准差: {sdf_std}")

    # 获取绝对路径
    liver_data_abs_path = os.path.abspath(LIVER_DATA_PATH)
    sdf_data_abs_path = os.path.abspath(SDF_DATA_PATH)

    # 创建配置文件内容
    config_content = f"""
seed: 42
lambda_geom: {TEST_PARAMS['lambda_geom']}
batch_size: {TEST_PARAMS['batch_size']}
total_batch_size: {TEST_PARAMS['batch_size']}
sample_q: {TEST_PARAMS['sample_q']}
inp_size: {TEST_PARAMS['inp_size']}
queue_size: {TEST_PARAMS['batch_size'] * 2}

train_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {liver_data_abs_path}, sdf_path: {sdf_data_abs_path}, repeat: 1}}}}
  wrapper: {{name: sr-gaussian, args: {{inp_size: {TEST_PARAMS['inp_size']}, sample_q: {TEST_PARAMS['sample_q']}, augment: false, scale: {TEST_PARAMS['scale']}}}}}

val_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {liver_data_abs_path}, sdf_path: {sdf_data_abs_path}}}}}
  wrapper: {{name: sr-gaussian, args: {{inp_size: {TEST_PARAMS['inp_size']}, sample_q: {TEST_PARAMS['sample_q']}, scale: {TEST_PARAMS['scale']}}}}}

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
    
    with open(REAL_CONFIG_PATH, 'w') as f:
        f.write(config_content)
    print(f"配置文件已创建: {REAL_CONFIG_PATH}")
    print("-" * 20)


def run_real_data_pipeline_test():
    """使用真实数据运行流水线测试"""
    print("\n--- 步骤2: 开始真实数据流水线测试 ---")
    
    with open(REAL_CONFIG_PATH, 'r') as f:
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
    
    batch = next(iter(train_loader))
    print("成功从加载器获取一个批次的真实数据。")
    for k, v in batch.items():
        if v is not None:
            batch[k] = v.to(device)
            print(f"  - Batch['{k}'] shape: {v.shape}")
        else:
            print(f"  - Batch['{k}']: None")
            
    lr = degrade_op(batch['inp'], scale=TEST_PARAMS['scale'], norm=True)
    p = {'lr': lr, 'gt': batch['gt'], 'cell': batch['cell'], 'coord': batch['coord'], 'scale': batch['scale'].type(torch.FloatTensor), 'gt_sdf': batch.get('gt_sdf')}
    lr, gt, cell, coord, scale, gt_sdf = pool(p)
    
    print("\n成功通过数据队列。")
    print(f"  - LR shape: {lr.shape}")
    print(f"  - GT shape: {gt.shape}")
    print(f"  - GT_SDF shape: {gt_sdf.shape if gt_sdf is not None else 'None'}")

    print("\n执行前向传播...")
    optimizer.zero_grad()
    model.train()
    pred_rgb = model(lr, coord, cell, state='train')
    
    pred_sdf = model.SR.sdf_pred
    
    print(f"  - Pred_RGB shape: {pred_rgb.shape}")
    if pred_sdf is not None:
        print(f"  - Pred_SDF shape: {pred_sdf.shape}")
    else:
        print("  - Pred_SDF: None")

    print("计算损失...")
    loss_sr = sr_criterion(pred_rgb, gt)
    
    total_loss = loss_sr
    loss_geom_val = 0.0
    
    if pred_sdf is not None and gt_sdf is not None:
        # 修复：直接在归一化空间计算SDF损失，避免反归一化的数值问题
        # 这样可以避免大数值导致的梯度爆炸问题
        loss_geom = geom_criterion(pred_sdf, gt_sdf)
        total_loss = total_loss + config['lambda_geom'] * loss_geom
        loss_geom_val = loss_geom.item()

    print(f"  - SR Loss: {loss_sr.item():.4f}")
    print(f"  - Geom Loss: {loss_geom_val:.4f}")
    print(f"  - Total Loss: {total_loss.item():.4f}")

    print("执行反向传播...")
    total_loss.backward()
    optimizer.step()
    print("优化器步进完成。")
    
    print("-" * 20)
    print("\n✅ 真实数据流水线测试成功！所有步骤均无错误执行。")
    print("-" * 20)

def cleanup_config_file():
    """删除生成的配置文件"""
    print("\n--- 步骤3: 清理临时配置文件 ---")
    if os.path.exists(REAL_CONFIG_PATH):
        try:
            os.remove(REAL_CONFIG_PATH)
            print(f"已删除: {REAL_CONFIG_PATH}")
        except OSError as e:
            print(f"删除文件 {REAL_CONFIG_PATH} 时出错: {e}")

def validate_data_consistency():
    """验证数据一致性"""
    print("\n--- 数据一致性验证 ---")
    
    # 检查liver_data中的图片数量
    liver_files = [f for f in os.listdir(LIVER_DATA_PATH) if f.endswith('.png')]
    print(f"Liver数据文件数量: {len(liver_files)}")
    
    # 检查SDF数据
    sdf_data = np.load(SDF_DATA_PATH)
    print(f"SDF数据形状: {sdf_data.shape}")
    print(f"SDF数据类型: {sdf_data.dtype}")
    print(f"SDF值范围: [{sdf_data.min():.4f}, {sdf_data.max():.4f}]")
    
    # 检查一张示例图片
    if liver_files:
        sample_img_path = os.path.join(LIVER_DATA_PATH, liver_files[0])
        sample_img = io.imread(sample_img_path)
        print(f"示例图片形状: {sample_img.shape}")
        print(f"示例图片数据类型: {sample_img.dtype}")
        print(f"示例图片值范围: [{sample_img.min()}, {sample_img.max()}]")
    
    print("-" * 20)

if __name__ == '__main__':
    try:
        validate_data_consistency()
        create_real_data_config()
        run_real_data_pipeline_test()
    except Exception as e:
        print(f"\n❌ 测试失败，出现错误: {e}")
        traceback.print_exc()
    finally:
        cleanup_config_file()
