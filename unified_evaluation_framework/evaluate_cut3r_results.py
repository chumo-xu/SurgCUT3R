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
        self.results_base = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r_inference_results copy_code_test"
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
            "dual_evaluation_dataset9_keyframe3",
            "dual_evaluation_dataset9_keyframe3test"
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
                'pred_depth_dir': f"{self.results_base}/{sequence_name}/combined_depth",
                'pred_pose_file': f"{self.results_base}/{sequence_name}/stitched_predicted_poses.npz",
                
                # GT数据路径 (使用 keyframe3 的真值) 
                'gt_depth_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_depths_{dataset_num}_{keyframe_num}.npz",
                'gt_pose_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_absolute_poses_{dataset_num}_{keyframe_num}_c2w.npz",
                
                # 输出路径
                'output_dir': f"{self.results_base}/{sequence_name}/unified_evaluation_results"
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
            'pred_depth_dir': f"{self.results_base}/{sequence_name}/combined_depth",
            'pred_pose_file': f"{self.results_base}/{sequence_name}/stitched_predicted_poses.npz",
            
            # GT数据路径  
            'gt_depth_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_depths_{dataset_num}_{keyframe_num}.npz",
            'gt_pose_file': f"{self.gt_base}/dataset{dataset_num}/keyframe{keyframe_num}/gt_absolute_poses_{dataset_num}_{keyframe_num}_c2w.npz",
            
            # 输出路径
            'output_dir': f"{self.results_base}/{sequence_name}/unified_evaluation_results"
        }
        
        return paths
    
    def validate_sequence_paths(self, paths):
        """
        验证序列路径是否存在
        
        Args:
            paths: 路径字典
            
        Returns:
            bool: 是否所有路径都存在
        """
        required_paths = [
            ('pred_depth_dir', paths['pred_depth_dir']),
            ('pred_pose_file', paths['pred_pose_file']),
            ('gt_depth_file', paths['gt_depth_file']),
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
            results = self.evaluator.evaluate_complete(
                gt_depth_path=paths['gt_depth_file'],
                pred_depth_path=paths['pred_depth_dir'],
                gt_pose_path=paths['gt_pose_file'],
                pred_pose_path=paths['pred_pose_file'],
                output_dir=paths['output_dir'],
                # 深度配置
                gt_depth_format='npz_file',      # GT是NPZ文件
                pred_depth_format='npy_dir',     # 预测是NPY目录
                gt_depth_unit='m',               # 确认单位: 深度数据统一为米
                pred_depth_unit='m',             # 确认单位: 深度数据统一为米
                # 位姿配置
                gt_pose_format='npz_file',       # GT是NPZ文件
                pred_pose_format='npz_file',     # 预测是NPZ文件
                gt_pose_unit='m',                # 确认单位: 位姿位移统一为米
                pred_pose_unit='m',              # 确认单位: 位姿位移统一为米
                # 评估配置
                depth_mode='both',               # 使用双重深度评估方法（全局+分段）
                # 可视化配置
                sequence_name=sequence_name
            )
            
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
        # 收集所有成功序列的指标
        depth_metrics = []
        gate_rmses = []
        late_rmse_means = []
        late_rmse_stds = []
        
        for seq_name, result in all_results.items():
            depth_eval = result['depth_evaluation']
            pose_eval = result['pose_evaluation']['pose_metrics']
            
            # 选择深度指标（优先使用分段评估结果）
            if 'depth_metrics_segmented' in depth_eval:
                depth_metrics_data = depth_eval['depth_metrics_segmented']
            else:
                depth_metrics_data = depth_eval['depth_metrics']
            
            # 深度指标
            depth_metrics.append({
                'sequence': seq_name,
                'abs_rel': depth_metrics_data['abs_rel']['mean'],
                'sq_rel': depth_metrics_data['sq_rel']['mean'],
                'rmse': depth_metrics_data['rmse']['mean'],
                'rmse_log': depth_metrics_data['rmse_log']['mean'],
                'a1': depth_metrics_data['a1']['mean'],
                'a2': depth_metrics_data['a2']['mean'],
                'a3': depth_metrics_data['a3']['mean']
            })
            
            # 位姿指标
            gate_rmses.append(pose_eval['gate']['gate_rmse'])
            late_rmse_means.append(pose_eval['late']['late_rmse_mean'])
            late_rmse_stds.append(pose_eval['late']['late_rmse_std'])
        
        # 计算平均指标
        if len(depth_metrics) > 0:
            avg_depth_metrics = {
                'abs_rel': np.mean([m['abs_rel'] for m in depth_metrics]),
                'sq_rel': np.mean([m['sq_rel'] for m in depth_metrics]),
                'rmse': np.mean([m['rmse'] for m in depth_metrics]),
                'rmse_log': np.mean([m['rmse_log'] for m in depth_metrics]),
                'a1': np.mean([m['a1'] for m in depth_metrics]),
                'a2': np.mean([m['a2'] for m in depth_metrics]),
                'a3': np.mean([m['a3'] for m in depth_metrics])
            }
            
            avg_pose_metrics = {
                'gate_rmse': np.mean(gate_rmses),
                'late_rmse_mean': np.mean(late_rmse_means),
                'late_rmse_std': np.mean(late_rmse_stds)
            }
        else:
            avg_depth_metrics = {}
            avg_pose_metrics = {}
        
        summary = {
            'evaluation_info': {
                'total_sequences': len(self.sequences),
                'successful_sequences': len(successful_sequences),
                'failed_sequences': len(failed_sequences),
                'success_rate': len(successful_sequences) / len(self.sequences)
            },
            'successful_sequences': successful_sequences,
            'failed_sequences': failed_sequences,
            'average_metrics': {
                'depth': avg_depth_metrics,
                'pose': avg_pose_metrics
            },
            'per_sequence_metrics': {
                'depth': depth_metrics,
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
            f.write("CUT3R模型评估报告 - 统一评估框架\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
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
            
            # 平均指标
            if summary['average_metrics']['depth']:
                depth_avg = summary['average_metrics']['depth']
                pose_avg = summary['average_metrics']['pose']
                
                f.write("🎯 平均评估结果:\n")
                f.write("-" * 50 + "\n")
                f.write("深度指标 (单位说明: rmse单位为米m):\n")
                f.write(f"  abs_rel:  {depth_avg['abs_rel']:.4f}\n")
                f.write(f"  sq_rel:   {depth_avg['sq_rel']:.4f}\n")
                f.write(f"  rmse:     {depth_avg['rmse']:.4f} m\n")
                f.write(f"  rmse_log: {depth_avg['rmse_log']:.4f}\n")
                f.write(f"  a1:       {depth_avg['a1']:.4f}\n")
                f.write(f"  a2:       {depth_avg['a2']:.4f}\n")
                f.write(f"  a3:       {depth_avg['a3']:.4f}\n\n")
                
                f.write("位姿指标:\n")
                f.write(f"  G-ATE RMSE: {pose_avg['gate_rmse']:.4f} m\n")
                f.write(f"  L-ATE RMSE: {pose_avg['late_rmse_mean']:.4f} ± {pose_avg['late_rmse_std']:.4f} m\n\n")
            
            # 每个序列的详细结果
            f.write("📊 各序列详细结果:\n")
            f.write("-" * 50 + "\n")
            for depth_metric in summary['per_sequence_metrics']['depth']:
                seq = depth_metric['sequence']
                f.write(f"{seq}:\n")
                f.write(f"  深度: abs_rel={depth_metric['abs_rel']:.4f}, rmse={depth_metric['rmse']:.4f}m, a1={depth_metric['a1']:.4f}\n")
                
                # 找到对应的位姿指标
                seq_idx = summary['successful_sequences'].index(seq)
                gate_rmse = summary['per_sequence_metrics']['pose']['gate_rmses'][seq_idx]
                late_mean = summary['per_sequence_metrics']['pose']['late_rmse_means'][seq_idx]
                late_std = summary['per_sequence_metrics']['pose']['late_rmse_stds'][seq_idx]
                f.write(f"  位姿: G-ATE={gate_rmse:.4f}m, L-ATE={late_mean:.4f}±{late_std:.4f}m\n\n")
    
    def _print_summary(self, summary):
        """打印评估总结"""
        print("\n" + "=" * 80)
        print("🎉 CUT3R模型评估完成 - 总结报告")
        print("=" * 80)
        
        info = summary['evaluation_info']
        print(f"📊 评估统计:")
        print(f"  总序列数: {info['total_sequences']}")
        print(f"  成功评估: {info['successful_sequences']}")
        print(f"  成功率: {info['success_rate']:.1%}")
        
        if info['failed_sequences'] > 0:
            print(f"  失败序列: {summary['failed_sequences']}")
        
        if summary['average_metrics']['depth']:
            depth_avg = summary['average_metrics']['depth']
            pose_avg = summary['average_metrics']['pose']
            
            print(f"\n🎯 平均性能指标:")
            print(f"  深度估计 (单位说明: rmse单位为米m):")
            print(f"    abs_rel:  {depth_avg['abs_rel']:.4f}")
            print(f"    sq_rel:   {depth_avg['sq_rel']:.4f}")
            print(f"    rmse:     {depth_avg['rmse']:.4f} m")
            print(f"    rmse_log: {depth_avg['rmse_log']:.4f}")
            print(f"    a1:       {depth_avg['a1']:.4f}")
            print(f"    a2:       {depth_avg['a2']:.4f}")
            print(f"    a3:       {depth_avg['a3']:.4f}")
            
            print(f"  位姿估计:")
            print(f"    G-ATE:    {pose_avg['gate_rmse']:.4f} m")
            print(f"    L-ATE:    {pose_avg['late_rmse_mean']:.4f} ± {pose_avg['late_rmse_std']:.4f} m")
            
            # 性能评级
            print(f"\n🏆 性能评级:")
            if depth_avg['a1'] > 0.8:
                depth_grade = "优秀 (S级)"
            elif depth_avg['a1'] > 0.6:
                depth_grade = "良好 (A级)"
            else:
                depth_grade = "有待改进 (B级)"
            print(f"  深度估计: {depth_grade} (a1={depth_avg['a1']:.3f})")
            
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
