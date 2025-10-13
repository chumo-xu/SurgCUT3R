"""
统一评估框架使用示例
演示如何使用统一评估框架评估不同模型的深度和位姿结果
"""

import os
import sys
import argparse

# 添加框架路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluators import UnifiedEvaluator


def example_depth_only_evaluation():
    """
    示例1: 仅深度评估
    """
    print("\n" + "="*60)
    print("示例1: 仅深度评估")
    print("="*60)
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        depth_min=1e-3,      # 最小深度 1mm
        depth_max=150.0,     # 最大深度 150m
        verbose=True
    )
    
    # 示例路径（需要根据实际情况修改）
    gt_depth_path = "/path/to/gt_depths.npz"
    pred_depth_path = "/path/to/predicted_depths/"
    
    try:
        results = evaluator.evaluate_depth_only(
            gt_depth_path=gt_depth_path,
            pred_depth_path=pred_depth_path,
            gt_format='npz_file',        # GT格式: NPZ文件
            pred_format='npy_dir',       # 预测格式: NPY文件目录
            gt_unit='m',                 # GT单位: 米
            pred_unit='m'                # 预测单位: 米
        )
        
        # 打印结果
        evaluator.depth_evaluator.print_results(results)
        
    except Exception as e:
        print(f"深度评估示例失败: {e}")
        print("请确保数据路径正确且数据格式符合要求")


def example_pose_only_evaluation():
    """
    示例2: 仅位姿评估
    """
    print("\n" + "="*60)
    print("示例2: 仅位姿评估")
    print("="*60)
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        pose_window_size=5,  # L-ATE窗口大小：5帧不重叠
        verbose=True
    )
    
    # 示例路径（需要根据实际情况修改）
    gt_pose_path = "/path/to/gt_poses.npz"
    pred_pose_path = "/path/to/predicted_poses/"
    
    try:
        results = evaluator.evaluate_pose_only(
            gt_pose_path=gt_pose_path,
            pred_pose_path=pred_pose_path,
            gt_format='npz_file',        # GT格式: NPZ文件
            pred_format='npz_dir',       # 预测格式: NPZ文件目录
            gt_unit='m',                 # GT位移单位: 米
            pred_unit='m'                # 预测位移单位: 米
        )
        
        # 打印结果
        evaluator.pose_evaluator.print_results(results)
        
    except Exception as e:
        print(f"位姿评估示例失败: {e}")
        print("请确保数据路径正确且数据格式符合要求")


def example_complete_evaluation():
    """
    示例3: 完整评估 (深度 + 位姿)
    """
    print("\n" + "="*60)
    print("示例3: 完整评估 (深度 + 位姿)")
    print("="*60)
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        depth_min=1e-3,      # 深度范围: 1mm ~ 150m
        depth_max=150.0,
        pose_window_size=5,  # L-ATE窗口: 5帧不重叠
        verbose=True
    )
    
    # 示例路径（需要根据实际情况修改）
    gt_depth_path = "/path/to/gt_depths.npz"
    pred_depth_path = "/path/to/predicted_depths/"
    gt_pose_path = "/path/to/gt_poses.npz"  
    pred_pose_path = "/path/to/predicted_poses/"
    output_dir = "/path/to/evaluation_results/"
    
    try:
        results = evaluator.evaluate_complete(
            gt_depth_path=gt_depth_path,
            pred_depth_path=pred_depth_path,
            gt_pose_path=gt_pose_path,
            pred_pose_path=pred_pose_path,
            output_dir=output_dir,
            # 深度数据格式
            gt_depth_format='npz_file',
            pred_depth_format='npy_dir',
            gt_depth_unit='m',
            pred_depth_unit='m',
            # 位姿数据格式
            gt_pose_format='npz_file',
            pred_pose_format='npz_dir',
            gt_pose_unit='m',
            pred_pose_unit='m'
        )
        
        # 打印完整结果
        evaluator.print_complete_results(results)
        
    except Exception as e:
        print(f"完整评估示例失败: {e}")
        print("请确保所有数据路径正确且数据格式符合要求")


def example_scared_dataset_evaluation():
    """
    示例4: SCARED数据集评估（基于现有代码路径）
    """
    print("\n" + "="*60)
    print("示例4: SCARED数据集评估")
    print("="*60)
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        depth_min=1e-3,
        depth_max=150.0,
        pose_window_size=5,
        verbose=True
    )
    
    # SCARED数据集示例路径（基于你提供的代码路径结构）
    base_path = "/hy-tmp/hy-tmp/CUT3R"
    
    # 注意：这些路径需要根据实际数据位置调整
    gt_depth_path = f"{base_path}/data/scared_gt_depths.npz"  # 假设的GT深度
    pred_depth_path = f"{base_path}/results/predicted_depths/"  # 假设的预测深度
    gt_pose_path = f"{base_path}/data/scared_gt_poses.npz"    # 假设的GT位姿
    pred_pose_path = f"{base_path}/results/predicted_poses/"  # 假设的预测位姿
    output_dir = f"{base_path}/evaluation_results/"
    
    try:
        results = evaluator.evaluate_complete(
            gt_depth_path=gt_depth_path,
            pred_depth_path=pred_depth_path,
            gt_pose_path=gt_pose_path,
            pred_pose_path=pred_pose_path,
            output_dir=output_dir,
            # 深度配置
            gt_depth_format='npz_file',
            pred_depth_format='npy_dir',
            gt_depth_unit='m',      # SCARED深度单位通常为mm，但框架内部统一为m
            pred_depth_unit='m',
            # 位姿配置
            gt_pose_format='npz_file',
            pred_pose_format='npz_dir',
            gt_pose_unit='m',       # SCARED位姿位移单位通常为mm，但框架内部统一为m
            pred_pose_unit='m'
        )
        
        evaluator.print_complete_results(results)
        print(f"\n📁 详细结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"SCARED数据集评估失败: {e}")
        print("请检查数据路径是否存在以及数据格式是否正确")


def example_different_formats():
    """
    示例5: 不同数据格式的适配示例
    """
    print("\n" + "="*60)
    print("示例5: 不同数据格式适配")
    print("="*60)
    
    evaluator = UnifiedEvaluator(verbose=True)
    
    # 展示不同格式的支持
    format_examples = {
        '深度数据格式': {
            'npy_dir': "包含.npy文件的目录，每个文件一帧深度",
            'npz_file': "单个.npz文件，包含所有帧深度",
            'tiff_dir': "包含.tiff文件的目录，适用于SCARED等数据集"
        },
        '位姿数据格式': {
            'npz_dir': "包含.npz文件的目录，每个文件一个位姿矩阵", 
            'npz_file': "单个.npz文件，包含所有位姿矩阵",
            'txt_file': "文本文件，每行16个元素(4x4矩阵)或12个元素(KITTI格式)",
            'json_file': "JSON格式文件，包含位姿数组"
        }
    }
    
    print("支持的数据格式:")
    for category, formats in format_examples.items():
        print(f"\n{category}:")
        for fmt, description in formats.items():
            print(f"  - {fmt}: {description}")
    
    # 格式检测示例
    example_paths = [
        "/path/to/depths.npz",
        "/path/to/depth_dir/",
        "/path/to/poses.txt",
        "/path/to/pose_dir/"
    ]
    
    print(f"\n格式自动检测示例:")
    for path in example_paths:
        try:
            if 'depth' in path:
                info = evaluator.depth_adapter.get_format_info(path)
            else:
                info = evaluator.pose_adapter.get_format_info(path)
            print(f"  {path}: {info.get('suggested_format', '未知格式')}")
        except:
            print(f"  {path}: 路径不存在（示例路径）")


def main():
    """
    主函数 - 运行示例
    """
    parser = argparse.ArgumentParser(description='统一评估框架使用示例')
    parser.add_argument('--example', type=str, 
                       choices=['depth', 'pose', 'complete', 'scared', 'formats', 'all'],
                       default='all',
                       help='运行特定示例')
    
    args = parser.parse_args()
    
    print("🚀 统一评估框架使用示例")
    print("✨ 支持深度和位姿评估的统一标准")
    print("📦 支持多种数据格式的自动适配")
    
    if args.example == 'depth' or args.example == 'all':
        example_depth_only_evaluation()
    
    if args.example == 'pose' or args.example == 'all':
        example_pose_only_evaluation()
    
    if args.example == 'complete' or args.example == 'all':
        example_complete_evaluation()
    
    if args.example == 'scared' or args.example == 'all':
        example_scared_dataset_evaluation()
    
    if args.example == 'formats' or args.example == 'all':
        example_different_formats()
    
    print("\n" + "="*80)
    print("📋 使用说明:")
    print("1. 修改示例中的路径为你的实际数据路径")
    print("2. 确保数据格式符合要求（详见适配器文档）")
    print("3. 根据需要调整评估参数（深度范围、窗口大小等）")
    print("4. 查看输出目录中的详细结果和可视化")
    print("="*80)


if __name__ == "__main__":
    main()



