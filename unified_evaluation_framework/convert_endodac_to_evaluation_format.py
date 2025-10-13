#!/usr/bin/env python3
"""
EndoDAC结果格式转换脚本
将EndoDAC的深度和位姿结果转换为CUT3R评估代码需要的格式
"""

import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


class EndoDACResultConverter:
    """EndoDAC结果转换器"""
    
    def __init__(self):
        # 数据路径
        self.endodac_depth_base = "/hy-tmp/hy-tmp/CUT3R/eval/endodac_results_scared/hy-tmp/hy-tmp/EndoDAC"
        self.endodac_pose_base = "/hy-tmp/endodac_evaluation_results/hy-tmp/hy-tmp/EndoDAC/EndoDACScaredPose"
        
        # 输出路径
        self.output_base = "/hy-tmp/hy-tmp/CUT3R/eval/endodac_results_scared/EndoDAC_results"
        
        # 数据映射：(sequence_name, depth_dir, pose_dir)
        self.sequence_mappings = [
            ("dual_evaluation_dataset8_keyframe0", "evaluation_results_dataset8_0", "80"),
            ("dual_evaluation_dataset8_keyframe1", "evaluation_results_dataset8_1", "81"),
            ("dual_evaluation_dataset8_keyframe2", "evaluation_results_dataset8_2", "82"), 
            ("dual_evaluation_dataset8_keyframe3", "evaluation_results_dataset8_3", "83"),
            ("dual_evaluation_dataset9_keyframe0", "evaluation_results_dataset9_0", "90"),
            ("dual_evaluation_dataset9_keyframe1", "evaluation_results_dataset9_1", "91"),
            ("dual_evaluation_dataset9_keyframe2", "evaluation_results_dataset9_2", "92"),
            ("dual_evaluation_dataset9_keyframe3", "evaluation_results_dataset9_3", "93"),
        ]
    
    def validate_input_paths(self):
        """验证输入路径是否存在"""
        print("🔍 验证输入路径...")
        
        missing_paths = []
        
        for seq_name, depth_dir, pose_dir in self.sequence_mappings:
            depth_path = os.path.join(self.endodac_depth_base, depth_dir, "npy_depth_results")
            pose_file = os.path.join(self.endodac_pose_base, pose_dir, "absolute_poses.npz")
            
            if not os.path.exists(depth_path):
                missing_paths.append(f"深度路径: {depth_path}")
            if not os.path.exists(pose_file):
                missing_paths.append(f"位姿文件: {pose_file}")
        
        if missing_paths:
            print("❌ 缺少以下路径:")
            for path in missing_paths:
                print(f"   {path}")
            return False
        
        print("✅ 所有输入路径验证成功")
        return True
    
    def copy_depth_files(self, source_dir, target_dir):
        """
        复制深度文件（保持EndoDAC原始格式）
        从 depth_000000.npy → depth_000000.npy
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # 获取所有深度文件
        depth_files = sorted([f for f in os.listdir(source_dir) if f.startswith('depth_') and f.endswith('.npy')])
        
        if len(depth_files) == 0:
            raise ValueError(f"在 {source_dir} 中未找到深度文件")
        
        print(f"  复制 {len(depth_files)} 个深度文件...")
        
        for depth_file in tqdm(depth_files, desc="  深度文件"):
            source_path = os.path.join(source_dir, depth_file)
            target_path = os.path.join(target_dir, depth_file)  # 保持原始文件名
            
            # 复制文件
            shutil.copy2(source_path, target_path)
        
        print(f"  ✅ 深度文件复制完成: {len(depth_files)} 个文件")
        return len(depth_files)
    
    def get_gt_pose_frames(self, seq_name):
        """获取GT位姿的帧数"""
        # 根据序列名称解析数据集和关键帧信息
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
        
        if not os.path.exists(gt_pose_file):
            raise ValueError(f"GT位姿文件不存在: {gt_pose_file}")
        
        gt_poses = np.load(gt_pose_file)
        return gt_poses['data'].shape[0]
    
    def convert_pose_file(self, source_file, target_file, gt_frames):
        """
        转换位姿文件格式并处理帧数问题
        从 absolute_poses.npz → stitched_predicted_poses.npz
        确保帧数与GT位姿一致
        """
        print(f"  转换位姿文件: {os.path.basename(source_file)}")
        
        # 加载原始位姿数据
        poses_data = np.load(source_file)
        poses = poses_data['data']
        
        print(f"  原始位姿帧数: {poses.shape[0]}")
        print(f"  GT位姿帧数: {gt_frames}")
        
        # 如果帧数多于GT，去掉最后几帧
        if poses.shape[0] > gt_frames:
            poses_trimmed = poses[:gt_frames]
            print(f"  ⚠️  去掉最后 {poses.shape[0] - gt_frames} 帧")
        elif poses.shape[0] == gt_frames:
            poses_trimmed = poses
            print(f"  ✅ 位姿帧数正确")
        else:
            raise ValueError(f"位姿帧数不足: {poses.shape[0]} < {gt_frames}")
        
        # 保存转换后的位姿
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        np.savez(target_file, data=poses_trimmed)
        
        print(f"  ✅ 位姿文件转换完成: {poses_trimmed.shape}")
        
        return poses_trimmed.shape[0]
    
    def convert_single_sequence(self, seq_name, depth_dir, pose_dir):
        """转换单个序列"""
        print(f"\n🔄 转换序列: {seq_name}")
        
        # 输入路径
        source_depth_dir = os.path.join(self.endodac_depth_base, depth_dir, "npy_depth_results")
        source_pose_file = os.path.join(self.endodac_pose_base, pose_dir, "absolute_poses.npz")
        
        # 输出路径
        target_seq_dir = os.path.join(self.output_base, seq_name)
        target_depth_dir = os.path.join(target_seq_dir, "combined_depth")
        target_pose_file = os.path.join(target_seq_dir, "stitched_predicted_poses.npz")
        
        try:
            # 获取GT位姿帧数
            print("  📏 获取GT位姿帧数...")
            gt_frames = self.get_gt_pose_frames(seq_name)
            print(f"  GT位姿帧数: {gt_frames}")
            
            # 复制深度文件（保持原始格式）
            print("  📂 复制深度文件...")
            depth_count = self.copy_depth_files(source_depth_dir, target_depth_dir)
            
            # 转换位姿文件（调整为GT帧数）
            print("  📍 转换位姿文件...")
            pose_count = self.convert_pose_file(source_pose_file, target_pose_file, gt_frames)
            
            # 报告结果
            print(f"  📊 结果: {depth_count} 深度文件, {pose_count} 位姿帧")
            print(f"✅ 序列 {seq_name} 转换完成")
            return True
            
        except Exception as e:
            print(f"❌ 序列 {seq_name} 转换失败: {e}")
            return False
    
    def convert_all_sequences(self):
        """转换所有序列"""
        print("🚀 开始转换EndoDAC结果格式")
        print("=" * 80)
        
        # 验证输入路径
        if not self.validate_input_paths():
            return False
        
        # 创建输出目录
        os.makedirs(self.output_base, exist_ok=True)
        print(f"📁 输出目录: {self.output_base}")
        
        # 转换统计
        successful = 0
        failed = 0
        
        # 逐个转换序列
        for seq_name, depth_dir, pose_dir in self.sequence_mappings:
            if self.convert_single_sequence(seq_name, depth_dir, pose_dir):
                successful += 1
            else:
                failed += 1
        
        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 EndoDAC结果格式转换完成")
        print("=" * 80)
        print(f"📊 转换统计:")
        print(f"  总序列数: {len(self.sequence_mappings)}")
        print(f"  成功转换: {successful}")
        print(f"  转换失败: {failed}")
        print(f"  成功率: {successful/len(self.sequence_mappings):.1%}")
        
        if successful > 0:
            print(f"\n📁 转换结果保存在: {self.output_base}")
            print("现在可以使用评估代码评估EndoDAC的结果了！")
        
        return successful == len(self.sequence_mappings)
    
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
            
            status = "✅" if depth_count > 0 and pose_exists else "❌"
            print(f"  {status} {seq}: {depth_count} 深度文件, 位姿文件 {'存在' if pose_exists else '缺失'}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EndoDAC结果格式转换工具')
    parser.add_argument('--list', action='store_true', help='列出已转换的序列')
    parser.add_argument('--sequence', type=str, help='只转换指定序列')
    
    args = parser.parse_args()
    
    converter = EndoDACResultConverter()
    
    if args.list:
        converter.list_converted_sequences()
    elif args.sequence:
        # 转换单个序列
        mappings = {seq: (depth, pose) for seq, depth, pose in converter.sequence_mappings}
        if args.sequence not in mappings:
            print(f"❌ 序列 '{args.sequence}' 不在支持列表中")
            print(f"支持的序列: {list(mappings.keys())}")
            return
        
        depth_dir, pose_dir = mappings[args.sequence]
        success = converter.convert_single_sequence(args.sequence, depth_dir, pose_dir)
        print(f"{'✅' if success else '❌'} 序列 {args.sequence} 转换{'成功' if success else '失败'}")
    else:
        # 转换所有序列
        converter.convert_all_sequences()


if __name__ == "__main__":
    main()
