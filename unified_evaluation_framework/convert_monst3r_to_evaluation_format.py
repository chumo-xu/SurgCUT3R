#!/usr/bin/env python3
"""
MonST3R结果格式转换脚本
将MonST3R的深度和位姿结果转换为CUT3R评估代码需要的格式
"""

import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将四元数转换为旋转矩阵
    
    Args:
        qx, qy, qz, qw: 四元数分量
    
    Returns:
        3x3旋转矩阵
    """
    # 归一化四元数
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def pose_to_matrix(qx, qy, qz, qw, tx, ty, tz):
    """
    将四元数+位移转换为4x4变换矩阵
    
    Args:
        qx, qy, qz, qw: 四元数
        tx, ty, tz: 位移
    
    Returns:
        4x4变换矩阵
    """
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    
    return T


class MonST3RResultConverter:
    """MonST3R结果转换器"""
    
    def __init__(self):
        # 输入路径
        self.monst3r_input_dir = "/hy-tmp/hy-tmp/CUT3R/eval/monst3r/hy-tmp/monst3r/demo_tmp"
        
        # 输出路径
        self.output_base = "/hy-tmp/hy-tmp/CUT3R/eval/monst3r/monst3r_results"
        
        # 目录名到序列名的映射（实际存在的6个）
        self.existing_mappings = {
            "Scared8_1_Left_Images": "dual_evaluation_dataset8_keyframe1",
            "Scared8_2_Left_Images": "dual_evaluation_dataset8_keyframe2", 
            "Scared8_3_Left_Images": "dual_evaluation_dataset8_keyframe3",
            "Scared9_0_Left_Images": "dual_evaluation_dataset9_keyframe0",
            "Scared9_1_Left_Images": "dual_evaluation_dataset9_keyframe1",
            "Scared9_3_Left_Images": "dual_evaluation_dataset9_keyframe3",
        }
        
        # 需要创建占位的序列（缺失的2个）
        self.missing_sequences = [
            "dual_evaluation_dataset8_keyframe0",
            "dual_evaluation_dataset9_keyframe2",
        ]
    
    def validate_input_dirs(self):
        """验证输入目录是否存在"""
        print("🔍 验证输入目录...")
        
        missing_dirs = []
        
        for dirname in self.existing_mappings.keys():
            dir_path = os.path.join(self.monst3r_input_dir, dirname)
            if not os.path.exists(dir_path):
                missing_dirs.append(dirname)
        
        if missing_dirs:
            print("❌ 缺少以下目录:")
            for dirname in missing_dirs:
                print(f"   {dirname}")
            return False
        
        print("✅ 所有输入目录验证成功")
        return True
    
    def parse_pose_file(self, pose_file):
        """
        解析位姿文件
        
        Args:
            pose_file: pred_traj.txt文件路径
        
        Returns:
            (N, 4, 4) 位姿矩阵数组
        """
        print(f"  解析位姿文件: {os.path.basename(pose_file)}")
        
        poses = []
        
        with open(pose_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    parts = line.strip().split()
                    if len(parts) != 8:
                        raise ValueError(f"第{line_num+1}行格式错误，期望8列，实际{len(parts)}列")
                    
                    # 解析：时间戳, qx, qy, qz, qw, tx, ty, tz
                    timestamp = float(parts[0])
                    qx, qy, qz, qw = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    
                    # 转换为4x4矩阵
                    pose_matrix = pose_to_matrix(qx, qy, qz, qw, tx, ty, tz)
                    poses.append(pose_matrix)
                    
                except Exception as e:
                    print(f"  ⚠️ 第{line_num+1}行解析失败: {e}")
                    continue
        
        poses_array = np.array(poses)
        print(f"  ✅ 位姿解析完成: {poses_array.shape}")
        
        return poses_array
    
    def copy_depth_files(self, source_dir, target_dir):
        """
        复制深度文件并重命名为标准格式
        从 frame_0000.npy → 000000.npy
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # 获取所有深度文件
        depth_files = sorted([f for f in os.listdir(source_dir) if f.startswith('frame_') and f.endswith('.npy')])
        
        if len(depth_files) == 0:
            raise ValueError(f"在 {source_dir} 中未找到深度文件")
        
        print(f"  复制 {len(depth_files)} 个深度文件...")
        
        for i, depth_file in enumerate(tqdm(depth_files, desc="  深度文件")):
            # 标准化编号：000000.npy, 000001.npy, ...
            target_filename = f"{i:06d}.npy"
            
            source_path = os.path.join(source_dir, depth_file)
            target_path = os.path.join(target_dir, target_filename)
            
            # 复制文件
            shutil.copy2(source_path, target_path)
        
        print(f"  ✅ 深度文件复制完成: {len(depth_files)} 个文件")
        return len(depth_files)
    
    def save_pose_data(self, poses_array, target_file):
        """
        保存位姿数据
        """
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        print(f"  保存位姿数据: {poses_array.shape}")
        
        # 保存为评估代码需要的格式
        np.savez(target_file, data=poses_array)
        
        print(f"  ✅ 位姿数据保存完成: {poses_array.shape}")
        return poses_array.shape[0]
    
    def get_gt_frames_count(self, seq_name):
        """获取GT数据的帧数（用于创建占位数据）"""
        # 解析序列名称
        if seq_name == "dual_evaluation_dataset9_keyframe3test":
            dataset_num = '9'
            keyframe_num = '3'
        else:
            parts = seq_name.split('_')
            dataset = parts[2]  # dataset8
            keyframe = parts[3] # keyframe0
            dataset_num = dataset.replace('dataset', '')  # 8
            keyframe_num = keyframe.replace('keyframe', '') # 0
        
        gt_pose_file = f"/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test/dataset{dataset_num}/keyframe{keyframe_num}/gt_absolute_poses_{dataset_num}_{keyframe_num}_c2w.npz"
        
        if os.path.exists(gt_pose_file):
            gt_poses = np.load(gt_pose_file)
            return gt_poses['data'].shape[0]
        else:
            # 如果找不到GT，使用默认值
            print(f"  ⚠️ 未找到GT文件，使用默认帧数")
            return 945  # 默认帧数
    
    def create_placeholder_sequence(self, seq_name):
        """为缺失的序列创建全0占位数据"""
        print(f"\n🔄 创建占位序列: {seq_name}")
        
        # 输出路径
        target_seq_dir = os.path.join(self.output_base, seq_name)
        target_depth_dir = os.path.join(target_seq_dir, "combined_depth")
        target_pose_file = os.path.join(target_seq_dir, "stitched_predicted_poses.npz")
        
        try:
            # 获取GT帧数
            gt_frames = self.get_gt_frames_count(seq_name)
            print(f"  创建 {gt_frames} 帧占位数据...")
            
            # 创建全0深度文件（192x240分辨率，与其他模型一致）
            os.makedirs(target_depth_dir, exist_ok=True)
            
            for i in tqdm(range(gt_frames), desc="  创建深度文件"):
                depth_filename = f"{i:06d}.npy"
                depth_path = os.path.join(target_depth_dir, depth_filename)
                
                # 创建全0深度图（使用合理的分辨率）
                zero_depth = np.zeros((192, 240), dtype=np.float32)
                np.save(depth_path, zero_depth)
            
            # 创建全0位姿文件（单位矩阵）
            identity_poses = np.tile(np.eye(4), (gt_frames, 1, 1))
            np.savez(target_pose_file, data=identity_poses)
            
            print(f"  ✅ 占位序列创建完成: {gt_frames} 帧")
            return True
            
        except Exception as e:
            print(f"  ❌ 占位序列创建失败: {e}")
            return False
    
    def convert_single_sequence(self, dirname, seq_name):
        """转换单个序列"""
        print(f"\n🔄 转换序列: {dirname} → {seq_name}")
        
        # 输入路径
        source_dir = os.path.join(self.monst3r_input_dir, dirname)
        source_pose_file = os.path.join(source_dir, "pred_traj.txt")
        
        # 输出路径
        target_seq_dir = os.path.join(self.output_base, seq_name)
        target_depth_dir = os.path.join(target_seq_dir, "combined_depth")
        target_pose_file = os.path.join(target_seq_dir, "stitched_predicted_poses.npz")
        
        try:
            # 解析位姿文件
            print("  📍 解析位姿数据...")
            poses_array = self.parse_pose_file(source_pose_file)
            
            # 复制深度文件
            print("  📂 复制深度文件...")
            depth_count = self.copy_depth_files(source_dir, target_depth_dir)
            
            # 保存位姿数据
            print("  💾 保存位姿数据...")
            pose_count = self.save_pose_data(poses_array, target_pose_file)
            
            # 验证帧数一致性
            if depth_count != pose_count:
                print(f"  ⚠️  警告: 深度帧数({depth_count}) != 位姿帧数({pose_count})")
            else:
                print(f"  ✅ 帧数一致: {depth_count} 帧")
            
            print(f"✅ 序列 {dirname} 转换完成")
            return True
            
        except Exception as e:
            print(f"❌ 序列 {dirname} 转换失败: {e}")
            return False
    
    def convert_all_sequences(self):
        """转换所有序列（包括创建占位序列）"""
        print("🚀 开始转换MonST3R结果格式")
        print("=" * 80)
        
        # 验证输入目录
        if not self.validate_input_dirs():
            return False
        
        # 创建输出目录
        os.makedirs(self.output_base, exist_ok=True)
        print(f"📁 输出目录: {self.output_base}")
        
        # 转换统计
        successful = 0
        failed = 0
        
        # 转换实际存在的6个序列
        print(f"\n📊 转换实际序列 ({len(self.existing_mappings)}个):")
        for dirname, seq_name in self.existing_mappings.items():
            if self.convert_single_sequence(dirname, seq_name):
                successful += 1
            else:
                failed += 1
        
        # 创建缺失序列的占位数据
        print(f"\n🔧 创建占位序列 ({len(self.missing_sequences)}个):")
        for seq_name in self.missing_sequences:
            if self.create_placeholder_sequence(seq_name):
                successful += 1
            else:
                failed += 1
        
        total_sequences = len(self.existing_mappings) + len(self.missing_sequences)
        
        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 MonST3R结果格式转换完成")
        print("=" * 80)
        print(f"📊 转换统计:")
        print(f"  总序列数: {total_sequences}")
        print(f"  实际数据: {len(self.existing_mappings)}")
        print(f"  占位数据: {len(self.missing_sequences)}")
        print(f"  成功转换: {successful}")
        print(f"  转换失败: {failed}")
        print(f"  成功率: {successful/total_sequences:.1%}")
        
        if successful > 0:
            print(f"\n📁 转换结果保存在: {self.output_base}")
            print("现在可以使用评估代码评估MonST3R的结果了！")
        
        return successful == total_sequences
    
    def list_converted_sequences(self):
        """列出已转换的序列"""
        if not os.path.exists(self.output_base):
            print("❌ 输出目录不存在，请先运行转换")
            return
        
        print(f"📁 已转换的序列 ({self.output_base}):")
        sequences = [d for d in os.listdir(self.output_base) 
                    if os.path.isdir(os.path.join(self.output_base, d))]
        
        for seq in sorted(sequences):
            seq_dir = os.path.join(self.output_base, seq)
            depth_dir = os.path.join(seq_dir, "combined_depth")
            pose_file = os.path.join(seq_dir, "stitched_predicted_poses.npz")
            
            depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.npy')]) if os.path.exists(depth_dir) else 0
            pose_exists = os.path.exists(pose_file)
            
            # 检查是否为占位数据
            is_placeholder = seq in self.missing_sequences
            seq_type = "占位" if is_placeholder else "实际"
            
            status = "✅" if depth_count > 0 and pose_exists else "❌"
            print(f"  {status} {seq}: {depth_count} 深度文件, 位姿文件 {'存在' if pose_exists else '缺失'} ({seq_type})")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MonST3R结果格式转换工具')
    parser.add_argument('--list', action='store_true', help='列出已转换的序列')
    parser.add_argument('--sequence', type=str, help='只转换指定序列目录')
    
    args = parser.parse_args()
    
    converter = MonST3RResultConverter()
    
    if args.list:
        converter.list_converted_sequences()
    elif args.sequence:
        # 转换单个序列
        if args.sequence not in converter.existing_mappings:
            print(f"❌ 序列 '{args.sequence}' 不在支持列表中")
            print(f"支持的序列: {list(converter.existing_mappings.keys())}")
            return
        
        seq_name = converter.existing_mappings[args.sequence]
        success = converter.convert_single_sequence(args.sequence, seq_name)
        print(f"{'✅' if success else '❌'} 序列 {args.sequence} 转换{'成功' if success else '失败'}")
    else:
        # 转换所有序列
        converter.convert_all_sequences()


if __name__ == "__main__":
    main()
