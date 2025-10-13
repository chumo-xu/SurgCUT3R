#!/usr/bin/env python3
"""
MegaSAM结果格式转换脚本
将MegaSAM的深度和位姿结果转换为CUT3R评估代码需要的格式
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm


class MegaSAMResultConverter:
    """MegaSAM结果转换器"""
    
    def __init__(self):
        # 输入路径
        self.megasam_input_dir = "/hy-tmp/hy-tmp/CUT3R/eval/megasam/hy-tmp/mega-sam/outputs_cvd"
        
        # 输出路径
        self.output_base = "/hy-tmp/hy-tmp/CUT3R/eval/megasam/megasam_results"
        
        # 文件名到序列名的映射
        self.file_mappings = {
            "Scared8_0_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset8_keyframe0",
            "Scared8_1_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset8_keyframe1", 
            "Scared8_2_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset8_keyframe2",
            "Scared8_3_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset8_keyframe3",
            "Scared9_0_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset9_keyframe0",
            "Scared9_1_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset9_keyframe1",
            "Scared9_2_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset9_keyframe2",
            "Scared9_3_Left_Images_sgd_cvd_hr.npz": "dual_evaluation_dataset9_keyframe3",
        }
    
    def validate_input_files(self):
        """验证输入文件是否存在"""
        print("🔍 验证输入文件...")
        
        missing_files = []
        
        for filename in self.file_mappings.keys():
            file_path = os.path.join(self.megasam_input_dir, filename)
            if not os.path.exists(file_path):
                missing_files.append(filename)
        
        if missing_files:
            print("❌ 缺少以下文件:")
            for filename in missing_files:
                print(f"   {filename}")
            return False
        
        print("✅ 所有输入文件验证成功")
        return True
    
    def extract_depth_data(self, depths_array, target_dir):
        """
        提取深度数据为单独的npy文件
        从 (N, H, W) → 000000.npy, 000001.npy, ...
        """
        os.makedirs(target_dir, exist_ok=True)
        
        num_frames = depths_array.shape[0]
        print(f"  提取 {num_frames} 帧深度数据...")
        
        for i in tqdm(range(num_frames), desc="  深度帧"):
            # 格式化帧号为6位数字
            frame_filename = f"{i:06d}.npy"
            frame_path = os.path.join(target_dir, frame_filename)
            
            # 保存单帧深度
            np.save(frame_path, depths_array[i])
        
        print(f"  ✅ 深度数据提取完成: {num_frames} 个文件")
        return num_frames
    
    def extract_pose_data(self, poses_array, target_file):
        """
        提取位姿数据
        从 cam_c2w (N, 4, 4) → stitched_predicted_poses.npz
        """
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        print(f"  保存位姿数据: {poses_array.shape}")
        
        # 保存为评估代码需要的格式
        np.savez(target_file, data=poses_array)
        
        print(f"  ✅ 位姿数据保存完成: {poses_array.shape}")
        return poses_array.shape[0]
    
    def convert_single_file(self, filename, seq_name):
        """转换单个MegaSAM结果文件"""
        print(f"\n🔄 转换文件: {filename} → {seq_name}")
        
        # 输入文件路径
        input_file = os.path.join(self.megasam_input_dir, filename)
        
        # 输出路径
        target_seq_dir = os.path.join(self.output_base, seq_name)
        target_depth_dir = os.path.join(target_seq_dir, "combined_depth")
        target_pose_file = os.path.join(target_seq_dir, "stitched_predicted_poses.npz")
        
        try:
            # 加载MegaSAM数据
            print("  📂 加载MegaSAM结果...")
            data = np.load(input_file)
            
            # 检查数据结构
            print(f"  数据键: {list(data.keys())}")
            depths = data['depths']  # (N, H, W)
            poses = data['cam_c2w']  # (N, 4, 4)
            
            print(f"  深度形状: {depths.shape}")
            print(f"  位姿形状: {poses.shape}")
            
            # 验证帧数一致性
            if depths.shape[0] != poses.shape[0]:
                raise ValueError(f"深度帧数({depths.shape[0]}) != 位姿帧数({poses.shape[0]})")
            
            # 提取深度数据
            print("  📊 提取深度数据...")
            depth_count = self.extract_depth_data(depths, target_depth_dir)
            
            # 提取位姿数据
            print("  📍 提取位姿数据...")
            pose_count = self.extract_pose_data(poses, target_pose_file)
            
            # 验证结果
            if depth_count != pose_count:
                print(f"  ⚠️  警告: 深度帧数({depth_count}) != 位姿帧数({pose_count})")
            else:
                print(f"  ✅ 帧数一致: {depth_count} 帧")
            
            print(f"✅ 文件 {filename} 转换完成")
            return True
            
        except Exception as e:
            print(f"❌ 文件 {filename} 转换失败: {e}")
            return False
    
    def convert_all_files(self):
        """转换所有MegaSAM结果文件"""
        print("🚀 开始转换MegaSAM结果格式")
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
        
        # 逐个转换文件
        for filename, seq_name in self.file_mappings.items():
            if self.convert_single_file(filename, seq_name):
                successful += 1
            else:
                failed += 1
        
        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 MegaSAM结果格式转换完成")
        print("=" * 80)
        print(f"📊 转换统计:")
        print(f"  总文件数: {len(self.file_mappings)}")
        print(f"  成功转换: {successful}")
        print(f"  转换失败: {failed}")
        print(f"  成功率: {successful/len(self.file_mappings):.1%}")
        
        if successful > 0:
            print(f"\n📁 转换结果保存在: {self.output_base}")
            print("现在可以使用评估代码评估MegaSAM的结果了！")
        
        return successful == len(self.file_mappings)
    
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
    
    parser = argparse.ArgumentParser(description='MegaSAM结果格式转换工具')
    parser.add_argument('--list', action='store_true', help='列出已转换的序列')
    parser.add_argument('--file', type=str, help='只转换指定文件')
    
    args = parser.parse_args()
    
    converter = MegaSAMResultConverter()
    
    if args.list:
        converter.list_converted_sequences()
    elif args.file:
        # 转换单个文件
        if args.file not in converter.file_mappings:
            print(f"❌ 文件 '{args.file}' 不在支持列表中")
            print(f"支持的文件: {list(converter.file_mappings.keys())}")
            return
        
        seq_name = converter.file_mappings[args.file]
        success = converter.convert_single_file(args.file, seq_name)
        print(f"{'✅' if success else '❌'} 文件 {args.file} 转换{'成功' if success else '失败'}")
    else:
        # 转换所有文件
        converter.convert_all_files()


if __name__ == "__main__":
    main()
