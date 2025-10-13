#!/usr/bin/env python3
"""
Spann3R结果格式转换脚本
将Spann3R的深度和位姿结果转换为CUT3R评估代码需要的格式
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


class Spann3RResultConverter:
    """Spann3R结果转换器"""
    
    def __init__(self):
        # 输入路径
        self.spann3r_input_dir = "/hy-tmp/hy-tmp/CUT3R/eval/spann3r/hy-tmp/output"
        
        # 输出路径
        self.output_base = "/hy-tmp/hy-tmp/CUT3R/eval/spann3r/spann3r_results"
        
        # 目录名到序列名的映射
        self.sequence_mappings = {
            "Scared8_0_Left_Images": "dual_evaluation_dataset8_keyframe0",
            "Scared8_1_Left_Images": "dual_evaluation_dataset8_keyframe1",
            "Scared8_2_Left_Images": "dual_evaluation_dataset8_keyframe2", 
            "Scared8_3_Left_Images": "dual_evaluation_dataset8_keyframe3",
            "Scared9_0_Left_Images": "dual_evaluation_dataset9_keyframe0",
            "Scared9_1_Left_Images": "dual_evaluation_dataset9_keyframe1",
            "Scared9_2_Left_Images": "dual_evaluation_dataset9_keyframe2",
            "Scared9_3_Left_Images": "dual_evaluation_dataset9_keyframe3",
        }
    
    def validate_input_files(self):
        """验证输入文件是否存在"""
        print("🔍 验证输入文件...")
        
        missing_files = []
        
        for dirname in self.sequence_mappings.keys():
            # spann3r的文件路径格式：dirname/dirname/dirname.npy
            file_path = os.path.join(self.spann3r_input_dir, dirname, dirname, f"{dirname}.npy")
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ 缺少以下文件:")
            for filepath in missing_files:
                print(f"   {filepath}")
            return False
        
        print("✅ 所有输入文件验证成功")
        return True
    
    def extract_depth_data(self, pts_all, target_dir):
        """
        从3D点云数据中提取深度信息
        从 pts_all[:, :, :, 2] → 000000.npy, 000001.npy, ...
        """
        os.makedirs(target_dir, exist_ok=True)
        
        # 提取深度数据（Z坐标）
        depth_data = pts_all[:, :, :, 2]  # (N, H, W)
        num_frames = depth_data.shape[0]
        
        print(f"  提取 {num_frames} 帧深度数据...")
        print(f"  深度分辨率: {depth_data.shape[1]}x{depth_data.shape[2]}")
        
        for i in tqdm(range(num_frames), desc="  深度帧"):
            # 格式化帧号为6位数字
            frame_filename = f"{i:06d}.npy"
            frame_path = os.path.join(target_dir, frame_filename)
            
            # 保存单帧深度
            np.save(frame_path, depth_data[i])
        
        print(f"  ✅ 深度数据提取完成: {num_frames} 个文件")
        return num_frames
    
    def extract_pose_data(self, poses_all, target_file):
        """
        提取位姿数据
        poses_all 已经是 (N, 4, 4) 格式，直接保存
        """
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        print(f"  保存位姿数据: {poses_all.shape}")
        
        # 保存为评估代码需要的格式
        np.savez(target_file, data=poses_all)
        
        print(f"  ✅ 位姿数据保存完成: {poses_all.shape}")
        return poses_all.shape[0]
    
    def convert_single_sequence(self, dirname, seq_name):
        """转换单个序列"""
        print(f"\n🔄 转换序列: {dirname} → {seq_name}")
        
        # 输入文件路径
        input_file = os.path.join(self.spann3r_input_dir, dirname, dirname, f"{dirname}.npy")
        
        # 输出路径
        target_seq_dir = os.path.join(self.output_base, seq_name)
        target_depth_dir = os.path.join(target_seq_dir, "combined_depth")
        target_pose_file = os.path.join(target_seq_dir, "stitched_predicted_poses.npz")
        
        try:
            # 加载Spann3R数据
            print("  📂 加载Spann3R结果...")
            data = np.load(input_file, allow_pickle=True).item()
            
            # 检查数据结构
            required_keys = ['pts_all', 'poses_all']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"缺少必需的数据键: {key}")
            
            pts_all = data['pts_all']      # (N, H, W, 3)
            poses_all = data['poses_all']  # (N, 4, 4)
            
            print(f"  3D点云形状: {pts_all.shape}")
            print(f"  位姿形状: {poses_all.shape}")
            
            # 验证帧数一致性
            if pts_all.shape[0] != poses_all.shape[0]:
                raise ValueError(f"点云帧数({pts_all.shape[0]}) != 位姿帧数({poses_all.shape[0]})")
            
            # 提取深度数据
            print("  📊 提取深度数据...")
            depth_count = self.extract_depth_data(pts_all, target_depth_dir)
            
            # 提取位姿数据
            print("  📍 提取位姿数据...")
            pose_count = self.extract_pose_data(poses_all, target_pose_file)
            
            # 验证结果
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
        """转换所有Spann3R结果序列"""
        print("🚀 开始转换Spann3R结果格式")
        print("=" * 80)
        
        # 验证输入文件
        if not self.validate_input_files():
            return False
        
        # 创建输出目录
        os.makedirs(self.output_base, exist_ok=True)
        print(f"📁 输出目录: {self.output_base}")
        
        # 转换统计
        successful = 0
        failed = 0
        
        # 逐个转换序列
        for dirname, seq_name in self.sequence_mappings.items():
            if self.convert_single_sequence(dirname, seq_name):
                successful += 1
            else:
                failed += 1
        
        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 Spann3R结果格式转换完成")
        print("=" * 80)
        print(f"📊 转换统计:")
        print(f"  总序列数: {len(self.sequence_mappings)}")
        print(f"  成功转换: {successful}")
        print(f"  转换失败: {failed}")
        print(f"  成功率: {successful/len(self.sequence_mappings):.1%}")
        
        if successful > 0:
            print(f"\n📁 转换结果保存在: {self.output_base}")
            print("现在可以使用评估代码评估Spann3R的结果了！")
        
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
    
    parser = argparse.ArgumentParser(description='Spann3R结果格式转换工具')
    parser.add_argument('--list', action='store_true', help='列出已转换的序列')
    parser.add_argument('--sequence', type=str, help='只转换指定序列目录')
    
    args = parser.parse_args()
    
    converter = Spann3RResultConverter()
    
    if args.list:
        converter.list_converted_sequences()
    elif args.sequence:
        # 转换单个序列
        if args.sequence not in converter.sequence_mappings:
            print(f"❌ 序列 '{args.sequence}' 不在支持列表中")
            print(f"支持的序列: {list(converter.sequence_mappings.keys())}")
            return
        
        seq_name = converter.sequence_mappings[args.sequence]
        success = converter.convert_single_sequence(args.sequence, seq_name)
        print(f"{'✅' if success else '❌'} 序列 {args.sequence} 转换{'成功' if success else '失败'}")
    else:
        # 转换所有序列
        converter.convert_all_sequences()


if __name__ == "__main__":
    main()
