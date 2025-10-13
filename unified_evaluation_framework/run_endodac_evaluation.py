#!/usr/bin/env python3
"""
EndoDAC评估运行脚本
简化的接口来运行EndoDAC模型评估
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from evaluate_endodac_results import EndoDACResultsEvaluator


def check_endodac_data_availability():
    """检查EndoDAC数据的可用性"""
    print("🔍 检查EndoDAC数据可用性...")
    
    base_path = "/hy-tmp/hy-tmp/CUT3R/eval/endodac_results/hy-tmp/hy-tmp/EndoDAC"
    
    # 检查基础目录
    if not os.path.exists(base_path):
        print(f"❌ EndoDAC基础目录不存在: {base_path}")
        return False
    
    # 检查深度结果目录
    depth_dirs = []
    pose_dirs = []
    
    for i in range(8):
        if i < 4:
            dataset = 8
            keyframe = i
        else:
            dataset = 9
            keyframe = i - 4
        
        # 深度目录
        depth_dir = f"{base_path}/evaluation_results_dataset{dataset}_{keyframe}/npy_depth_results"
        depth_dirs.append((f"dataset{dataset}_{keyframe}", depth_dir))
        
        # 位姿文件
        pose_dir_num = f"{dataset}{keyframe}"
        pose_file = f"{base_path}/EndoDACScaredPose/{pose_dir_num}/absolute_poses.npz"
        pose_dirs.append((f"dataset{dataset}_{keyframe}", pose_file))
    
    print("\n📁 深度结果检查:")
    depth_available = 0
    for name, path in depth_dirs:
        if os.path.exists(path):
            # 检查是否有npy文件
            npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
            if len(npy_files) > 0:
                print(f"  ✅ {name}: {len(npy_files)} 个文件 - {path}")
                depth_available += 1
            else:
                print(f"  ❌ {name}: 目录存在但无npy文件 - {path}")
        else:
            print(f"  ❌ {name}: 目录不存在 - {path}")
    
    print("\n📍 位姿结果检查:")
    pose_available = 0
    for name, path in pose_dirs:
        if os.path.exists(path):
            print(f"  ✅ {name}: 文件存在 - {path}")
            pose_available += 1
        else:
            print(f"  ❌ {name}: 文件不存在 - {path}")
    
    print(f"\n📊 可用性总结:")
    print(f"  深度数据: {depth_available}/8 可用")
    print(f"  位姿数据: {pose_available}/8 可用")
    
    return depth_available > 0 and pose_available > 0


def run_endodac_evaluation():
    """运行EndoDAC评估"""
    print("🚀 开始EndoDAC模型评估")
    print("=" * 60)
    
    # 检查数据可用性
    if not check_endodac_data_availability():
        print("\n❌ 数据检查失败，无法继续评估")
        print("请确保EndoDAC结果数据已正确放置在指定路径")
        return False
    
    print("\n✅ 数据检查通过，开始评估...")
    
    try:
        # 初始化评估器
        evaluator = EndoDACResultsEvaluator(verbose=True)
        
        # 运行评估
        results = evaluator.evaluate_all_sequences()
        
        if len(results) > 0:
            print(f"\n🎉 EndoDAC评估成功完成！")
            print(f"成功评估了 {len(results)} 个序列")
            return True
        else:
            print(f"\n❌ 评估失败，没有成功的序列")
            return False
            
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_single_sequence_evaluation(sequence_name):
    """评估单个序列"""
    print(f"🔍 评估单个EndoDAC序列: {sequence_name}")
    
    try:
        # 初始化评估器
        evaluator = EndoDACResultsEvaluator(verbose=True)
        
        # 查找序列信息
        sequence_info = None
        for seq in evaluator.sequences:
            if seq["name"] == sequence_name:
                sequence_info = seq
                break
        
        if sequence_info is None:
            print(f"❌ 序列 '{sequence_name}' 不在支持的序列列表中")
            print(f"支持的序列: {[seq['name'] for seq in evaluator.sequences]}")
            return False
        
        # 运行单个序列评估
        result = evaluator.evaluate_single_sequence(sequence_info)
        
        if result is not None:
            print(f"✅ 序列 {sequence_name} 评估成功")
            return True
        else:
            print(f"❌ 序列 {sequence_name} 评估失败")
            return False
            
    except Exception as e:
        print(f"❌ 评估序列 {sequence_name} 时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EndoDAC评估运行工具')
    parser.add_argument('--check-only', action='store_true', help='仅检查数据可用性，不进行评估')
    parser.add_argument('--sequence', type=str, help='评估特定序列')
    parser.add_argument('--list-sequences', action='store_true', help='列出所有支持的序列')
    
    args = parser.parse_args()
    
    if args.list_sequences:
        # 列出所有序列
        evaluator = EndoDACResultsEvaluator(verbose=False)
        print("支持的EndoDAC评估序列:")
        for seq in evaluator.sequences:
            print(f"  - {seq['name']} (dataset{seq['dataset_num']}_keyframe{seq['keyframe_num']})")
        return
    
    if args.check_only:
        # 仅检查数据
        check_endodac_data_availability()
        return
    
    if args.sequence:
        # 评估单个序列
        success = run_single_sequence_evaluation(args.sequence)
        if success:
            print("✅ 单序列评估完成")
        else:
            sys.exit(1)
    else:
        # 评估所有序列
        success = run_endodac_evaluation()
        if success:
            print("✅ 完整评估完成")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()


