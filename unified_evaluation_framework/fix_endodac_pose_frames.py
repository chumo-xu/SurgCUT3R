#!/usr/bin/env python3
"""
修复EndoDAC位姿数据帧数不匹配问题
将EndoDAC位姿数据从(N, 4, 4)修改为(N-1, 4, 4)来匹配GT深度数据帧数
"""

import numpy as np
import os
from pathlib import Path

def fix_pose_frames():
    """修复所有EndoDAC序列的位姿帧数"""
    
    # 序列列表
    sequences = [
        'dual_evaluation_dataset8_keyframe0',
        'dual_evaluation_dataset8_keyframe1', 
        'dual_evaluation_dataset8_keyframe2',
        'dual_evaluation_dataset8_keyframe3',
        'dual_evaluation_dataset9_keyframe0',
        'dual_evaluation_dataset9_keyframe1',
        'dual_evaluation_dataset9_keyframe2',
        'dual_evaluation_dataset9_keyframe3'
    ]
    
    # 数据集映射
    dataset_mapping = {
        'dual_evaluation_dataset8_keyframe0': ('8', '0'),
        'dual_evaluation_dataset8_keyframe1': ('8', '1'),
        'dual_evaluation_dataset8_keyframe2': ('8', '2'), 
        'dual_evaluation_dataset8_keyframe3': ('8', '3'),
        'dual_evaluation_dataset9_keyframe0': ('9', '0'),
        'dual_evaluation_dataset9_keyframe1': ('9', '1'),
        'dual_evaluation_dataset9_keyframe2': ('9', '2'),
        'dual_evaluation_dataset9_keyframe3': ('9', '3')
    }
    
    base_path = '/hy-tmp/hy-tmp/CUT3R/eval'
    
    print("🔧 开始修复EndoDAC位姿数据帧数...")
    print("=" * 60)
    
    for seq in sequences:
        print(f"\n📋 处理序列: {seq}")
        
        dataset, keyframe = dataset_mapping[seq]
        
        # GT深度文件路径
        gt_depth_file = f"{base_path}/eval_data/test/dataset{dataset}/keyframe{keyframe}/gt_depths_{dataset}_{keyframe}.npz"
        
        # EndoDAC位姿文件路径
        endodac_pose_file = f"{base_path}/endodac_cut3r_format/{seq}/stitched_predicted_poses.npz"
        
        try:
            # 读取GT深度数据获取目标帧数
            gt_depth_data = np.load(gt_depth_file)
            target_frames = gt_depth_data['data'].shape[0]
            
            # 读取EndoDAC位姿数据
            endodac_pose_data = np.load(endodac_pose_file)
            current_poses = endodac_pose_data['data']
            current_frames = current_poses.shape[0]
            
            print(f"   GT深度帧数: {target_frames}")
            print(f"   当前位姿帧数: {current_frames}")
            
            if current_frames == target_frames:
                print("   ✅ 帧数已匹配，无需修改")
                continue
            elif current_frames == target_frames + 1:
                # 去掉最后一帧
                fixed_poses = current_poses[:-1]
                print(f"   🔧 去掉最后一帧: {current_frames} -> {fixed_poses.shape[0]}")
            elif current_frames == target_frames - 1:
                print("   ❌ EndoDAC帧数少于GT，无法修复")
                continue
            else:
                print(f"   ❌ 帧数差异过大: {current_frames} vs {target_frames}")
                continue
            
            # 备份原文件
            backup_file = endodac_pose_file + '.backup'
            if not os.path.exists(backup_file):
                os.system(f"cp '{endodac_pose_file}' '{backup_file}'")
                print(f"   💾 已备份原文件: {backup_file}")
            
            # 保存修复后的位姿数据
            np.savez_compressed(endodac_pose_file, data=fixed_poses)
            print(f"   ✅ 已修复位姿文件: {fixed_poses.shape}")
            
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 EndoDAC位姿数据帧数修复完成！")
    
    # 验证修复结果
    print("\n📊 验证修复结果:")
    print("-" * 40)
    
    for seq in sequences:
        dataset, keyframe = dataset_mapping[seq]
        
        try:
            gt_depth_file = f"{base_path}/eval_data/test/dataset{dataset}/keyframe{keyframe}/gt_depths_{dataset}_{keyframe}.npz"
            endodac_pose_file = f"{base_path}/endodac_cut3r_format/{seq}/stitched_predicted_poses.npz"
            
            gt_depth_data = np.load(gt_depth_file)
            gt_frames = gt_depth_data['data'].shape[0]
            
            endodac_pose_data = np.load(endodac_pose_file)
            endodac_frames = endodac_pose_data['data'].shape[0]
            
            status = "✅ 匹配" if gt_frames == endodac_frames else "❌ 不匹配"
            print(f"{seq}: GT={gt_frames}, EndoDAC={endodac_frames} {status}")
            
        except Exception as e:
            print(f"{seq}: ❌ 验证失败 - {e}")

if __name__ == "__main__":
    fix_pose_frames()


