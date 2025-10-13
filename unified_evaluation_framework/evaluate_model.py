#!/usr/bin/env python3
"""
统一评估框架命令行工具
提供便捷的命令行接口进行模型评估
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加框架路径
framework_dir = Path(__file__).parent
sys.path.insert(0, str(framework_dir))

from evaluators import UnifiedEvaluator
from adapters import DepthAdapter, PoseAdapter


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def auto_detect_format(adapter, path: str) -> str:
    """自动检测数据格式"""
    info = adapter.get_format_info(path)
    if 'suggested_format' in info:
        return info['suggested_format']
    else:
        raise ValueError(f"无法自动检测格式: {path}")


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='统一评估框架 - 深度和位姿评估命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整评估
  python evaluate_model.py complete \\
    --gt-depth gt_depths.npz --pred-depth pred_depths/ \\
    --gt-pose gt_poses.npz --pred-pose pred_poses/ \\
    --output results/

  # 仅深度评估
  python evaluate_model.py depth \\
    --gt-depth gt_depths.npz --pred-depth pred_depths/ \\
    --output results/

  # 仅位姿评估
  python evaluate_model.py pose \\
    --gt-pose gt_poses.npz --pred-pose pred_poses/ \\
    --output results/

  # 使用配置文件
  python evaluate_model.py complete --config my_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='评估模式')
    
    # 完整评估
    complete_parser = subparsers.add_parser('complete', help='完整评估 (深度+位姿)')
    add_complete_arguments(complete_parser)
    
    # 仅深度评估
    depth_parser = subparsers.add_parser('depth', help='仅深度评估')
    add_depth_arguments(depth_parser)
    
    # 仅位姿评估
    pose_parser = subparsers.add_parser('pose', help='仅位姿评估')
    add_pose_arguments(pose_parser)
    
    # 信息查询
    info_parser = subparsers.add_parser('info', help='显示框架信息')
    info_parser.add_argument('--formats', action='store_true', help='显示支持的格式')
    
    return parser


def add_common_arguments(parser):
    """添加通用参数"""
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--depth-min', type=float, default=1e-3, help='最小有效深度 (m)')
    parser.add_argument('--depth-max', type=float, default=150.0, help='最大有效深度 (m)')
    parser.add_argument('--pose-window', type=int, default=5, help='位姿L-ATE窗口大小')


def add_depth_arguments(parser):
    """添加深度评估参数"""
    add_common_arguments(parser)
    parser.add_argument('--gt-depth', type=str, required=True, help='GT深度数据路径')
    parser.add_argument('--pred-depth', type=str, required=True, help='预测深度数据路径')
    parser.add_argument('--gt-depth-format', type=str, help='GT深度格式 (自动检测)')
    parser.add_argument('--pred-depth-format', type=str, help='预测深度格式 (自动检测)')
    parser.add_argument('--gt-depth-unit', type=str, default='m', choices=['mm', 'm'], help='GT深度单位')
    parser.add_argument('--pred-depth-unit', type=str, default='m', choices=['mm', 'm'], help='预测深度单位')


def add_pose_arguments(parser):
    """添加位姿评估参数"""
    add_common_arguments(parser)
    parser.add_argument('--gt-pose', type=str, required=True, help='GT位姿数据路径')
    parser.add_argument('--pred-pose', type=str, required=True, help='预测位姿数据路径')
    parser.add_argument('--gt-pose-format', type=str, help='GT位姿格式 (自动检测)')
    parser.add_argument('--pred-pose-format', type=str, help='预测位姿格式 (自动检测)')
    parser.add_argument('--gt-pose-unit', type=str, default='m', choices=['mm', 'm'], help='GT位姿位移单位')
    parser.add_argument('--pred-pose-unit', type=str, default='m', choices=['mm', 'm'], help='预测位姿位移单位')


def add_complete_arguments(parser):
    """添加完整评估参数"""
    add_common_arguments(parser)
    # 深度参数
    parser.add_argument('--gt-depth', type=str, required=True, help='GT深度数据路径')
    parser.add_argument('--pred-depth', type=str, required=True, help='预测深度数据路径')
    parser.add_argument('--gt-depth-format', type=str, help='GT深度格式')
    parser.add_argument('--pred-depth-format', type=str, help='预测深度格式')
    parser.add_argument('--gt-depth-unit', type=str, default='m', choices=['mm', 'm'], help='GT深度单位')
    parser.add_argument('--pred-depth-unit', type=str, default='m', choices=['mm', 'm'], help='预测深度单位')
    # 位姿参数
    parser.add_argument('--gt-pose', type=str, required=True, help='GT位姿数据路径')
    parser.add_argument('--pred-pose', type=str, required=True, help='预测位姿数据路径')
    parser.add_argument('--gt-pose-format', type=str, help='GT位姿格式')
    parser.add_argument('--pred-pose-format', type=str, help='预测位姿格式')
    parser.add_argument('--gt-pose-unit', type=str, default='m', choices=['mm', 'm'], help='GT位姿位移单位')
    parser.add_argument('--pred-pose-unit', type=str, default='m', choices=['mm', 'm'], help='预测位姿位移单位')


def run_depth_evaluation(args):
    """运行深度评估"""
    print("🎯 开始深度评估...")
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        verbose=args.verbose
    )
    
    # 自动检测格式
    if not args.gt_depth_format:
        args.gt_depth_format = auto_detect_format(evaluator.depth_adapter, args.gt_depth)
        print(f"自动检测GT深度格式: {args.gt_depth_format}")
    
    if not args.pred_depth_format:
        args.pred_depth_format = auto_detect_format(evaluator.depth_adapter, args.pred_depth)
        print(f"自动检测预测深度格式: {args.pred_depth_format}")
    
    # 执行评估
    results = evaluator.evaluate_depth_only(
        gt_depth_path=args.gt_depth,
        pred_depth_path=args.pred_depth,
        gt_format=args.gt_depth_format,
        pred_format=args.pred_depth_format,
        gt_unit=args.gt_depth_unit,
        pred_unit=args.pred_depth_unit
    )
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, 'depth_evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印结果
    evaluator.depth_evaluator.print_results(results)
    print(f"\n📄 详细结果已保存到: {results_path}")


def run_pose_evaluation(args):
    """运行位姿评估"""
    print("🎯 开始位姿评估...")
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        pose_window_size=args.pose_window,
        verbose=args.verbose
    )
    
    # 自动检测格式
    if not args.gt_pose_format:
        args.gt_pose_format = auto_detect_format(evaluator.pose_adapter, args.gt_pose)
        print(f"自动检测GT位姿格式: {args.gt_pose_format}")
    
    if not args.pred_pose_format:
        args.pred_pose_format = auto_detect_format(evaluator.pose_adapter, args.pred_pose)
        print(f"自动检测预测位姿格式: {args.pred_pose_format}")
    
    # 执行评估
    results = evaluator.evaluate_pose_only(
        gt_pose_path=args.gt_pose,
        pred_pose_path=args.pred_pose,
        gt_format=args.gt_pose_format,
        pred_format=args.pred_pose_format,
        gt_unit=args.gt_pose_unit,
        pred_unit=args.pred_pose_unit
    )
    
    # 保存结果
    os.makedirs(args.output, exist_ok=True)
    results_path = os.path.join(args.output, 'pose_evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 打印结果
    evaluator.pose_evaluator.print_results(results)
    print(f"\n📄 详细结果已保存到: {results_path}")


def run_complete_evaluation(args):
    """运行完整评估"""
    print("🚀 开始完整评估 (深度 + 位姿)...")
    
    # 初始化评估器
    evaluator = UnifiedEvaluator(
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        pose_window_size=args.pose_window,
        verbose=args.verbose
    )
    
    # 自动检测格式
    if not args.gt_depth_format:
        args.gt_depth_format = auto_detect_format(evaluator.depth_adapter, args.gt_depth)
    if not args.pred_depth_format:
        args.pred_depth_format = auto_detect_format(evaluator.depth_adapter, args.pred_depth)
    if not args.gt_pose_format:
        args.gt_pose_format = auto_detect_format(evaluator.pose_adapter, args.gt_pose)
    if not args.pred_pose_format:
        args.pred_pose_format = auto_detect_format(evaluator.pose_adapter, args.pred_pose)
    
    print(f"检测到格式: GT深度({args.gt_depth_format}), 预测深度({args.pred_depth_format})")
    print(f"检测到格式: GT位姿({args.gt_pose_format}), 预测位姿({args.pred_pose_format})")
    
    # 执行评估
    results = evaluator.evaluate_complete(
        gt_depth_path=args.gt_depth,
        pred_depth_path=args.pred_depth,
        gt_pose_path=args.gt_pose,
        pred_pose_path=args.pred_pose,
        output_dir=args.output,
        gt_depth_format=args.gt_depth_format,
        pred_depth_format=args.pred_depth_format,
        gt_pose_format=args.gt_pose_format,
        pred_pose_format=args.pred_pose_format,
        gt_depth_unit=args.gt_depth_unit,
        pred_depth_unit=args.pred_depth_unit,
        gt_pose_unit=args.gt_pose_unit,
        pred_pose_unit=args.pred_pose_unit
    )
    
    # 打印结果
    evaluator.print_complete_results(results)
    print(f"\n📁 所有结果已保存到: {args.output}")


def show_framework_info(args):
    """显示框架信息"""
    print("=" * 60)
    print("🚀 统一评估框架 v1.0.0")
    print("=" * 60)
    print("📝 深度估计和位姿估计的统一评估框架")
    print("\n✨ 主要特性:")
    print("  • 标准化深度评估流程")
    print("  • 统一位姿评估标准 (G-ATE + L-ATE)")
    print("  • 多格式数据自动适配")
    print("  • 完整结果输出和可视化")
    print("\n📦 支持格式:")
    print("  深度: npy_dir, npz_file, tiff_dir")
    print("  位姿: npz_dir, npz_file, txt_file, json_file")
    
    if args.formats:
        print("\n📋 详细格式支持:")
        print("\n深度数据格式:")
        depth_formats = {
            'npy_dir': "NPY文件目录，每个文件一帧深度",
            'npz_file': "单个NPZ文件，包含所有帧深度",
            'tiff_dir': "TIFF文件目录，适用于SCARED等数据集"
        }
        for fmt, desc in depth_formats.items():
            print(f"  • {fmt}: {desc}")
        
        print("\n位姿数据格式:")
        pose_formats = {
            'npz_dir': "NPZ文件目录，每个文件一个4x4位姿矩阵",
            'npz_file': "单个NPZ文件，包含所有位姿",
            'txt_file': "文本文件，每行16个元素或12个元素(KITTI)",
            'json_file': "JSON格式文件"
        }
        for fmt, desc in pose_formats.items():
            print(f"  • {fmt}: {desc}")


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    try:
        if args.mode == 'depth':
            run_depth_evaluation(args)
        elif args.mode == 'pose':
            run_pose_evaluation(args)
        elif args.mode == 'complete':
            run_complete_evaluation(args)
        elif args.mode == 'info':
            show_framework_info(args)
        else:
            print(f"未知模式: {args.mode}")
            parser.print_help()
    
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
