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

# --- 1. 定义测试参数和文件路径 (SDF文件改为.npy) ---
DUMMY_CONFIG_PATH = 'configs/test_pipeline.yaml'
DUMMY_IMG_PATH_TRAIN = 'dummy_train_img.tif'
DUMMY_SDF_PATH_TRAIN = 'dummy_train_sdf.npy' # <--- 修改
DUMMY_IMG_PATH_VAL = 'dummy_val_img.tif'
DUMMY_SDF_PATH_VAL = 'dummy_val_sdf.npy'   # <--- 修改
CONFIG_DIR = 'configs'

TEST_PARAMS = {
    'img_shape': (16, 64, 64),
    'batch_size': 2,
    'inp_size': 24, 
    'sample_q': 1024,
    'scale': 2,
    'lambda_geom': 0.1
}

# --- 2. 创建虚拟文件 ---
def create_dummy_files():
    """生成虚拟的.tif数据文件和.yaml配置文件"""
    print("--- 步骤1: 创建虚拟数据和配置文件 ---")
    
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)

    dummy_image_train = np.random.randint(0, 256, size=TEST_PARAMS['img_shape']).astype(np.uint8)
    io.imsave(DUMMY_IMG_PATH_TRAIN, dummy_image_train)
    
    # 使用 np.save 创建 .npy 文件
    print(f"创建训练SDF: {DUMMY_SDF_PATH_TRAIN}")
    dummy_sdf_train = np.random.randn(*TEST_PARAMS['img_shape']).astype(np.float32)
    np.save(DUMMY_SDF_PATH_TRAIN, dummy_sdf_train) # <--- 修改

    dummy_image_val = np.random.randint(0, 256, size=TEST_PARAMS['img_shape']).astype(np.uint8)
    io.imsave(DUMMY_IMG_PATH_VAL, dummy_image_val)
    
    print(f"创建验证SDF: {DUMMY_SDF_PATH_VAL}")
    dummy_sdf_val = np.random.randn(*TEST_PARAMS['img_shape']).astype(np.float32)
    np.save(DUMMY_SDF_PATH_VAL, dummy_sdf_val) # <--- 修改
    
    sdf_mean = np.mean(dummy_sdf_train)
    sdf_std = np.std(dummy_sdf_train)

    # 在配置中指向 .npy 文件
    config_content = f"""
seed: 42
lambda_geom: {TEST_PARAMS['lambda_geom']}
batch_size: {TEST_PARAMS['batch_size']}
total_batch_size: {TEST_PARAMS['batch_size']}
sample_q: {TEST_PARAMS['sample_q']}
inp_size: {TEST_PARAMS['inp_size']}
queue_size: {TEST_PARAMS['batch_size'] * 2}

train_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {DUMMY_IMG_PATH_TRAIN}, sdf_path: {DUMMY_SDF_PATH_TRAIN}, repeat: 1}}}}
  wrapper: {{name: sr-gaussian, args: {{inp_size: {TEST_PARAMS['inp_size']}, sample_q: {TEST_PARAMS['sample_q']}, augment: false, scale: {TEST_PARAMS['scale']}}}}}

val_dataset1:
  dataset: {{name: image-volume, args: {{root_path: {DUMMY_IMG_PATH_VAL}, sdf_path: {DUMMY_SDF_PATH_VAL}}}}}
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
    
    with open(DUMMY_CONFIG_PATH, 'w') as f:
        f.write(config_content)
    print(f"配置文件已创建: {DUMMY_CONFIG_PATH}")
    print("-" * 20)


# --- 后续的主测试逻辑和清理逻辑保持不变 ---
def run_pipeline_test():
    """加载虚拟配置和数据，并运行一个训练步骤"""
    print("\n--- 步骤2: 开始流水线测试 ---")
    
    with open(DUMMY_CONFIG_PATH, 'r') as f:
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
    print("成功从加载器获取一个批次的数据。")
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
    print("\n✅ 流水线测试成功！所有步骤均无错误执行。")
    print("-" * 20)

def cleanup_dummy_files():
    """删除所有生成的虚拟文件"""
    print("\n--- 步骤3: 清理临时文件 ---")
    files_to_remove = [
        DUMMY_CONFIG_PATH, DUMMY_IMG_PATH_TRAIN, DUMMY_SDF_PATH_TRAIN,
        DUMMY_IMG_PATH_VAL, DUMMY_SDF_PATH_VAL
    ]
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"已删除: {f}")
            except OSError as e:
                print(f"删除文件 {f} 时出错: {e}")

if __name__ == '__main__':
    try:
        create_dummy_files()
        run_pipeline_test()
    except Exception as e:
        print(f"\n❌ 测试失败，出现错误: {e}")
        traceback.print_exc()
    finally:
        cleanup_dummy_files()