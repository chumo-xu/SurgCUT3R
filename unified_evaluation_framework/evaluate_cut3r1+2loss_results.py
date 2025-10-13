#!/usr/bin/env python3
"""
CUT3R结果评估脚本
使用统一评估框架评估CUT3R的推理结果
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

# 添加框架路径
framework_dir = Path(__file__).parent
sys.path.insert(0, str(framework_dir))

from evaluators.unified_evaluator import UnifiedEvaluator


class CUT3RResultsEvaluator:
    """CUT3R结果评估器"""
    
    def __init__(self, verbose=True):
        """
        初始化CUT3R结果评估器
        
        Args:
            verbose: 是否显示详细信息
        """
        self.verbose = verbose
        
        # CUT3R结果路径
        self.results_base = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss"
        self.gt_base = "/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test"
        
        # 序列列表
        self.sequences = [
            "dual_evaluation_dataset8_keyframe0",
            "dual_evaluation_dataset8_keyframe1", 
            "dual_evaluation_dataset8_keyframe2",
            "dual_evaluation_dataset8_keyframe3",
            "dual_evaluation_dataset9_keyframe0",
            "dual_evaluation_dataset9_keyframe1",
            "dual_evaluation_dataset9_keyframe2",
            "dual_evaluation_dataset9_keyframe3"
        ]
        
        # 初始化统一评估器
        self.evaluator = UnifiedEvaluator(
            depth_min=1e-3,      # 最小深度 1mm
            depth_max=150.0,     # 最大深度 150m 
            pose_window_size=16, # L-ATE窗口大小
            verbose=verbose
        )
        
        if verbose:
            print("🚀 CUT3R结果评估器初始化完成")
            print(f"评估序列数: {len(self.sequences)}")
    
    def get_sequence_paths(self, sequence_name):
        """
        获取单个序列的所有路径
        
        Args:
            sequence_name: 序列名称，如 "dual_evaluation_dataset8_keyframe0"
        
        Returns:
            dict: 包含所有路径的字典
        """
        # --- 为测试序列添加的临时修改 ---
        if sequence_name == "dual_evaluation_dataset9_keyframe3test":
            dataset_num = '9'
            keyframe_num = '3' # 使用原始keyframe3的真值进行比较
            paths = {
                'sequence_name': sequence_name,
                'dataset_num': dataset_num,
                'keyframe_num': keyframe_num,
                
                # 预测结果路径 (来自用户指定的测试文件夹)
                'pred_pose_file': f"/hy-tmp/hy-tmp/CUT3R/eval/test_all_sequences/{sequence_name}/cut3r1+cut3r2.npz",
                
                # GT数据路径 (使用 keyframe3 的真值) 
                'gt_pose_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_absolute_poses_{dataset_num}_{keyframe_num}_c2w.npz",
                
                # 输出路径
                'output_dir': f"{self.results_base}/{sequence_name}/unified_evaluation_results_pose_only"
            }
            return paths
        # --- 临时修改结束 ---

        # 解析序列名称
        # dual_evaluation_dataset8_keyframe0 -> dataset8, keyframe0
        parts = sequence_name.split('_')
        dataset = parts[2]  # dataset8
        keyframe = parts[3] # keyframe0
        
        # 提取数字
        dataset_num = dataset.replace('dataset', '')  # 8
        keyframe_num = keyframe.replace('keyframe', '') # 0
        
        paths = {
            'sequence_name': sequence_name,
            'dataset_num': dataset_num,
            'keyframe_num': keyframe_num,
            
            # 预测结果路径
            'pred_pose_file': f"/hy-tmp/hy-tmp/CUT3R/eval/test_all_sequences/{sequence_name}/cut3r1+cut3r2.npz",
            
            # GT数据路径  
            'gt_pose_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_absolute_poses_{dataset_num}_{keyframe_num}_c2w.npz",
            
            # 输出路径
            'output_dir': f"{self.results_base}/{sequence_name}/unified_evaluation_results_pose_only"
        }
        
        return paths
    
    def validate_sequence_paths(self, paths):
        """
        验证序列路径是否存在
        """
        required_paths = [
            ('pred_pose_file', paths['pred_pose_file']),
            ('gt_pose_file', paths['gt_pose_file'])
        ]
        
        missing_paths = []
        for name, path in required_paths:
            if not os.path.exists(path):
                missing_paths.append((name, path))
        
        if missing_paths:
            print(f"❌ 序列 {paths['sequence_name']} 缺少路径:")
            for name, path in missing_paths:
                print(f"   {name}: {path}")
            return False
        
        return True
    
    def evaluate_single_sequence(self, sequence_name):
        """
        评估单个序列
        
        Args:
            sequence_name: 序列名称
            
        Returns:
            dict: 评估结果，如果失败返回None
        """
        if self.verbose:
            print(f"\n🔍 评估序列: {sequence_name}")
        
        # 获取路径
        paths = self.get_sequence_paths(sequence_name)
        
        # 验证路径
        if not self.validate_sequence_paths(paths):
            return None
        
        try:
            # 使用统一评估框架进行评估
            pose_results = self.evaluator.evaluate_pose_only(
                gt_pose_path=paths['gt_pose_file'],
                pred_pose_path=paths['pred_pose_file'],
                output_dir=paths['output_dir'],
                # 位姿配置
                gt_pose_format='npz_file',       # GT是NPZ文件
                pred_pose_format='npz_file',     # <--- 关键修正：显式指定预测位姿为单个文件
                gt_pose_unit='m',                # 确认单位: 位姿位移统一为米
                pred_pose_unit='m',              # 确认单位: 位姿位移统一为米
                # 可视化配置
                sequence_name=sequence_name
            )

            # 将 `evaluate_pose_only` 的直接结果包装成报告函数期望的结构
            results = {'pose_evaluation': pose_results}
            
            if self.verbose:
                print(f"✅ 序列 {sequence_name} 评估成功")
            
            return results
            
        except Exception as e:
            print(f"❌ 序列 {sequence_name} 评估失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def evaluate_all_sequences(self, output_summary_dir=None):
        """
        评估所有序列
        
        Args:
            output_summary_dir: 总结果输出目录，如果为None则使用默认目录
        
        Returns:
            dict: 所有序列的评估结果
        """
        if output_summary_dir is None:
            output_summary_dir = f"{self.results_base}/unified_evaluation_summary"
        
        os.makedirs(output_summary_dir, exist_ok=True)
        
        print("🚀 开始评估所有CUT3R序列")
        print("=" * 70)
        
        all_results = {}
        successful_sequences = []
        failed_sequences = []
        
        # 逐个序列评估
        for sequence_name in tqdm(self.sequences, desc="评估序列"):
            result = self.evaluate_single_sequence(sequence_name)
            
            if result is not None:
                all_results[sequence_name] = result
                successful_sequences.append(sequence_name)
            else:
                failed_sequences.append(sequence_name)
        
        # 生成总结报告
        summary = self._generate_summary_report(all_results, successful_sequences, failed_sequences)
        
        # 保存总结果
        summary_file = os.path.join(output_summary_dir, 'cut3r_evaluation_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细报告
        report_file = os.path.join(output_summary_dir, 'cut3r_evaluation_report.txt')
        self._save_detailed_report(summary, report_file)
        
        # 打印总结
        self._print_summary(summary)
        
        print(f"\n📁 详细结果已保存到: {output_summary_dir}")
        
        return all_results
    
    def _generate_summary_report(self, all_results, successful_sequences, failed_sequences):
        """生成总结报告"""
        gate_rmses = []
        late_rmse_means = []
        late_rmse_stds = []
        
        for seq_name, result in all_results.items():
            pose_eval = result['pose_evaluation']['pose_metrics']
            
            gate_rmses.append(pose_eval['gate']['gate_rmse'])
            late_rmse_means.append(pose_eval['late']['late_rmse_mean'])
            late_rmse_stds.append(pose_eval['late']['late_rmse_std'])
        
        if len(gate_rmses) > 0:
            avg_pose_metrics = {
                'gate_rmse': np.mean(gate_rmses),
                'late_rmse_mean': np.mean(late_rmse_means),
                'late_rmse_std': np.mean(late_rmse_stds)
            }
        else:
            avg_pose_metrics = {}
        
        summary = {
            'evaluation_info': {
                'total_sequences': len(self.sequences),
                'successful_sequences': len(successful_sequences),
                'failed_sequences': len(failed_sequences),
                'success_rate': len(successful_sequences) / len(self.sequences) if len(self.sequences) > 0 else 0
            },
            'successful_sequences': successful_sequences,
            'failed_sequences': failed_sequences,
            'average_metrics': {
                'pose': avg_pose_metrics
            },
            'per_sequence_metrics': {
                'pose': {
                    'gate_rmses': gate_rmses,
                    'late_rmse_means': late_rmse_means,
                    'late_rmse_stds': late_rmse_stds
                }
            }
        }
        
        return summary
    
    def _save_detailed_report(self, summary, report_file):
        """保存详细文本报告"""
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CUT3R模型评估报告 - 统一评估框架 (仅位姿)\n")
            f.write("=" * 80 + "\n\n")
            
            info = summary['evaluation_info']
            f.write(f"评估序列总数: {info['total_sequences']}\n")
            f.write(f"成功评估: {info['successful_sequences']}\n")
            f.write(f"失败评估: {info['failed_sequences']}\n")
            f.write(f"成功率: {info['success_rate']:.1%}\n\n")
            
            if info['failed_sequences'] > 0:
                f.write("失败序列:\n")
                for seq in summary['failed_sequences']:
                    f.write(f"  - {seq}\n")
                f.write("\n")
            
            if summary['average_metrics']['pose']:
                pose_avg = summary['average_metrics']['pose']
                
                f.write("🎯 平均评估结果:\n")
                f.write("-" * 50 + "\n")
                f.write("位姿指标:\n")
                f.write(f"  G-ATE RMSE: {pose_avg['gate_rmse']:.4f} m\n")
                f.write(f"  L-ATE RMSE: {pose_avg['late_rmse_mean']:.4f} ± {pose_avg['late_rmse_std']:.4f} m\n\n")
            
            f.write("📊 各序列详细结果:\n")
            f.write("-" * 50 + "\n")
            for i, seq in enumerate(summary['successful_sequences']):
                f.write(f"{seq}:\n")
                gate_rmse = summary['per_sequence_metrics']['pose']['gate_rmses'][i]
                late_mean = summary['per_sequence_metrics']['pose']['late_rmse_means'][i]
                late_std = summary['per_sequence_metrics']['pose']['late_rmse_stds'][i]
                f.write(f"  位姿: G-ATE={gate_rmse:.4f}m, L-ATE={late_mean:.4f}±{late_std:.4f}m\n\n")
    
    def _print_summary(self, summary):
        """打印评估总结"""
        print("\n" + "=" * 80)
        print("🎉 CUT3R模型评估完成 - 总结报告 (仅位姿)")
        print("=" * 80)
        
        info = summary['evaluation_info']
        print(f"📊 评估统计:")
        print(f"  总序列数: {info['total_sequences']}")
        print(f"  成功评估: {info['successful_sequences']}")
        print(f"  成功率: {info['success_rate']:.1%}")
        
        if info['failed_sequences'] > 0:
            print(f"  失败序列: {summary['failed_sequences']}")
        
        if summary['average_metrics']['pose']:
            pose_avg = summary['average_metrics']['pose']
            
            print(f"\n🎯 平均性能指标:")
            print(f"  位姿估计:")
            print(f"    G-ATE:    {pose_avg['gate_rmse']:.4f} m")
            print(f"    L-ATE:    {pose_avg['late_rmse_mean']:.4f} ± {pose_avg['late_rmse_std']:.4f} m")
            
            print(f"\n🏆 性能评级:")
            if pose_avg['gate_rmse'] < 0.01 and pose_avg['late_rmse_mean'] < 0.01:
                pose_grade = "优秀 (S级)"
            elif pose_avg['gate_rmse'] < 0.05 and pose_avg['late_rmse_mean'] < 0.05:
                pose_grade = "良好 (A级)"
            else:
                pose_grade = "有待改进 (B级)"
            print(f"  位姿估计: {pose_grade}")
        
        print("=" * 80)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CUT3R结果评估工具')
    parser.add_argument('--sequence', type=str, help='评估特定序列，如果不指定则评估所有序列')
    parser.add_argument('--output', type=str, help='输出目录')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = CUT3RResultsEvaluator(verbose=args.verbose)
    
    if args.sequence:
        # 评估单个序列
        if args.sequence not in evaluator.sequences:
            print(f"❌ 序列 '{args.sequence}' 不在支持的序列列表中")
            print(f"支持的序列: {evaluator.sequences}")
            return
        
        result = evaluator.evaluate_single_sequence(args.sequence)
        if result:
            print(f"✅ 序列 {args.sequence} 评估完成")
            # 打印简要结果
            evaluator.evaluator.print_complete_results(result)
    else:
        # 评估所有序列
        all_results = evaluator.evaluate_all_sequences(args.output)
        print(f"✅ 所有序列评估完成，成功评估 {len(all_results)} 个序列")


if __name__ == "__main__":
    main()
