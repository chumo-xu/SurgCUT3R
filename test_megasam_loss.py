#!/usr/bin/env python3
"""
测试MegaSAMIntegratedLoss的完整功能
创建模拟数据来验证损失函数是否能正确运行
"""

import torch
import numpy as np
import sys
import os

# 添加src路径
sys.path.append('/hy-tmp/hy-tmp/CUT3R/src')

def create_mock_data(batch_size=2, num_views=4, H=128, W=128):
    """
    创建模拟的CUT3R格式数据
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建gts (ground truth数据)
    gts = []
    for view_idx in range(num_views):
        gt = {
            'img': torch.randn(batch_size, 3, H, W, device=device),  # 图像数据
            'camera_intrinsics': torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1),  # 内参
            'pts3d': torch.randn(batch_size, H, W, 3, device=device),  # 3D点
            'valid_mask': torch.ones(batch_size, H, W, dtype=torch.bool, device=device),  # 有效掩码
        }
        # 设置固定的内参（适应新尺寸）
        gt['camera_intrinsics'][:, 0, 0] = 200.0  # fx
        gt['camera_intrinsics'][:, 1, 1] = 200.0  # fy
        gt['camera_intrinsics'][:, 0, 2] = W / 2  # cx
        gt['camera_intrinsics'][:, 1, 2] = H / 2  # cy
        gts.append(gt)
    
    # 创建preds (预测数据)
    preds = []
    for view_idx in range(num_views):
        pred = {
            'pts3d_in_self_view': torch.randn(batch_size, H, W, 3, device=device),  # 自视角3D点
            'camera_pose': torch.randn(batch_size, 7, device=device),  # 位姿编码 (3位置+4四元数)
            'conf_self': torch.sigmoid(torch.randn(batch_size, H, W, device=device)),  # 置信度
            'conf': torch.sigmoid(torch.randn(batch_size, H, W, device=device)),  # 交叉视图置信度
        }
        
        # 确保深度为正值（合理范围）
        pred['pts3d_in_self_view'][..., 2] = torch.rand(batch_size, H, W, device=device) * 9.0 + 1.0  # 1-10米
        
        # 归一化四元数
        pred['camera_pose'][:, 3:] = torch.nn.functional.normalize(pred['camera_pose'][:, 3:], dim=1)
        
        # 确保需要梯度
        pred['pts3d_in_self_view'].requires_grad_(True)
        pred['camera_pose'].requires_grad_(True)
        
        preds.append(pred)
    
    return gts, preds

def test_megasam_loss():
    """
    测试MegaSAMIntegratedLoss的完整功能
    """
    print("=" * 60)
    print("开始测试MegaSAMIntegratedLoss")
    print("=" * 60)
    
    try:
        # 1. 导入损失函数
        from dust3r.losses import MegaSAMIntegratedLoss
        print("✅ 成功导入MegaSAMIntegratedLoss")
        
        # 2. 创建损失函数实例
        loss_fn = MegaSAMIntegratedLoss(
            w_megasam=0.1, 
            temporal_steps=[1, 2]  # 使用较小的步长进行测试
        )
        print(f"✅ 成功创建损失函数: {loss_fn.get_name()}")
        
        # 3. 创建模拟数据
        print("\n创建模拟数据...")
        gts, preds = create_mock_data(batch_size=2, num_views=4, H=128, W=128)
        print(f"✅ 创建了 {len(gts)} 个视图的模拟数据")
        
        # 4. 测试序列提取
        print("\n测试序列提取...")
        sequence_data = loss_fn.extract_sequence(gts, preds, batch_idx=0)
        print(f"✅ 序列提取成功，包含 {sequence_data['num_views']} 个视图")
        print(f"   - 图像形状: {sequence_data['images'][0].shape}")
        print(f"   - 3D点形状: {sequence_data['pts3d'][0].shape}")
        print(f"   - 位姿形状: {sequence_data['camera_poses'][0].shape}")
        
        # 5. 测试光流计算
        print("\n测试光流计算...")
        flows, flow_masks, ii, jj = loss_fn.compute_sequence_flows(sequence_data)
        print(f"✅ 光流计算成功")
        print(f"   - 光流形状: {flows.shape}")
        print(f"   - 光流掩码形状: {flow_masks.shape}")
        print(f"   - 帧索引对数: {len(ii)}")
        
        # 6. 测试数据格式转换
        print("\n测试数据格式转换...")
        megasam_inputs = loss_fn.convert_to_megasam_format(sequence_data, flows, flow_masks, ii, jj)
        print(f"✅ 数据格式转换成功")
        print(f"   - cam_c2w形状: {megasam_inputs['cam_c2w'].shape}")
        print(f"   - disp_data形状: {megasam_inputs['disp_data'].shape}")
        print(f"   - K矩阵形状: {megasam_inputs['K'].shape}")
        
        # 7. 测试完整损失计算
        print("\n测试完整损失计算...")
        loss, details = loss_fn.compute_loss(gts, preds)
        print(f"✅ 损失计算成功!")
        print(f"   - 损失值: {loss.item():.6f}")
        print(f"   - 损失梯度: {loss.requires_grad}")
        print(f"   - 详细信息: {details}")
        
        # 8. 测试梯度反向传播
        print("\n测试梯度反向传播...")
        loss.backward()
        print("✅ 梯度反向传播成功!")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过! MegaSAMIntegratedLoss可以正常使用!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_megasam_loss()
    exit(0 if success else 1)
