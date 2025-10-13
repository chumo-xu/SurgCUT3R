"""
统一评估器 - 深度和位姿评估的统一入口
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from core.depth_evaluator import DepthEvaluator
from core.pose_evaluator import PoseEvaluator
from core import utils
from core.visualization import plot_trajectory_3d_evaluation, plot_detailed_pose_analysis, plot_depth_comparison
from adapters.depth_adapter import DepthAdapter
from adapters.pose_adapter import PoseAdapter


class UnifiedEvaluator:
    """
    统一评估器 - 提供深度和位姿评估的统一接口
    
    特性:
    - 自动数据格式适配
    - 标准化评估流程
    - 统一结果输出
    - 完整的可视化和报告
    """
    
    def __init__(self, 
                 depth_min: float = 1e-3, 
                 depth_max: float = 150.0,
                 pose_window_size: int = 16,
                 verbose: bool = True):
        """
        初始化统一评估器
        
        Args:
            depth_min: 深度有效范围最小值 (m)
            depth_max: 深度有效范围最大值 (m) 
            pose_window_size: 位姿L-ATE窗口大小（不重叠）
            verbose: 是否打印详细信息
        """
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.pose_window_size = pose_window_size
        self.verbose = verbose
        
        # 初始化子模块
        self.depth_adapter = DepthAdapter(verbose=verbose)
        self.pose_adapter = PoseAdapter(verbose=verbose)
        self.depth_evaluator = DepthEvaluator(
            min_depth=depth_min, 
            max_depth=depth_max, 
            verbose=verbose
        )
        self.pose_evaluator = PoseEvaluator(
            window_size=pose_window_size, 
            verbose=verbose
        )
        
        utils.log_evaluation_info("统一评估器初始化完成", verbose)
        utils.log_evaluation_info(f"深度范围: {depth_min:.1e}~{depth_max:.1f}m, 位姿窗口: {pose_window_size}", verbose)
    
    def evaluate_depth_only(self, 
                           gt_depth_path: str,
                           pred_depth_path: str,
                           gt_format: str = 'npz_file',
                           pred_format: str = 'npy_dir',
                           gt_unit: str = 'm',
                           pred_unit: str = 'm',
                           depth_mode: str = 'both',
                           **format_kwargs) -> Dict[str, Any]:
        """
        仅进行深度评估
        
        Args:
            gt_depth_path: GT深度数据路径
            pred_depth_path: 预测深度数据路径
            gt_format: GT数据格式
            pred_format: 预测数据格式
            gt_unit: GT深度单位 ('mm' 或 'm')
            pred_unit: 预测深度单位 ('mm' 或 'm')
            depth_mode: 深度评估模式 ('global'=全局中值缩放, 'segmented'=分段中值缩放, 'both'=两种方法)
            **format_kwargs: 格式特定参数
        
        Returns:
            深度评估结果
        """
        utils.log_evaluation_info("🎯 开始深度专项评估", self.verbose)
        
        # 1. 加载GT深度数据
        utils.log_evaluation_info("📂 加载GT深度数据...", self.verbose)
        gt_depths = self.depth_adapter.load(
            gt_depth_path, gt_format, gt_unit, **format_kwargs.get('gt_kwargs', {})
        )
        
        # 2. 加载预测深度数据
        utils.log_evaluation_info("📂 加载预测深度数据...", self.verbose)
        pred_depths = self.depth_adapter.load(
            pred_depth_path, pred_format, pred_unit, **format_kwargs.get('pred_kwargs', {})
        )
        
        # 3. 数据验证
        gt_validation = self.depth_adapter.validate_data(gt_depths)
        pred_validation = self.depth_adapter.validate_data(pred_depths)
        
        if not gt_validation['valid']:
            raise ValueError(f"GT深度数据验证失败: {gt_validation['error']}")
        if not pred_validation['valid']:
            raise ValueError(f"预测深度数据验证失败: {pred_validation['error']}")
        
        # 4. 深度评估
        utils.log_evaluation_info("🔍 执行深度评估...", self.verbose)
        depth_results = self.depth_evaluator.evaluate(gt_depths, pred_depths, mode=depth_mode)
        
        # 5. 添加输入信息
        depth_results['input_info'] = {
            'gt_path': gt_depth_path,
            'pred_path': pred_depth_path,
            'gt_format': gt_format,
            'pred_format': pred_format,
            'gt_unit': gt_unit,
            'pred_unit': pred_unit,
            'gt_validation': gt_validation,
            'pred_validation': pred_validation
        }
        
        utils.log_evaluation_info("✅ 深度评估完成", self.verbose)
        return depth_results
    
    def evaluate_pose_only(self,
                          gt_pose_path: str,
                          pred_pose_path: str,
                          gt_format: str = 'npz_file',
                          pred_format: str = 'npz_file', 
                          gt_unit: str = 'm',
                          pred_unit: str = 'm',
                          output_dir: str = None,
                          sequence_name: str = "sequence",
                          **format_kwargs) -> Dict[str, Any]:
        """
        仅进行位姿评估
        
        Args:
            gt_pose_path: GT位姿数据路径
            pred_pose_path: 预测位姿数据路径
            gt_format: GT数据格式
            pred_format: 预测数据格式
            gt_unit: GT位姿位移单位 ('mm' 或 'm')
            pred_unit: 预测位姿位移单位 ('mm' 或 'm')
            **format_kwargs: 格式特定参数
        
        Returns:
            位姿评估结果
        """
        utils.log_evaluation_info("🎯 开始位姿专项评估", self.verbose)
        
        # 1. 加载GT位姿数据
        utils.log_evaluation_info("📂 加载GT位姿数据...", self.verbose)
        gt_poses = self.pose_adapter.load(
            gt_pose_path, gt_format, gt_unit, **format_kwargs.get('gt_kwargs', {})
        )
        
        # 2. 加载预测位姿数据
        utils.log_evaluation_info("📂 加载预测位姿数据...", self.verbose)
        pred_poses = self.pose_adapter.load(
            pred_pose_path, pred_format, pred_unit, **format_kwargs.get('pred_kwargs', {})
        )
        
        # 3. 数据验证
        gt_validation = self.pose_adapter.validate_poses(gt_poses)
        pred_validation = self.pose_adapter.validate_poses(pred_poses)
        
        if not gt_validation['valid']:
            raise ValueError(f"GT位姿数据验证失败: {gt_validation}")
        if not pred_validation['valid']:
            raise ValueError(f"预测位姿数据验证失败: {pred_validation}")
        
        # 4. 位姿评估
        utils.log_evaluation_info("🔍 执行位姿评估...", self.verbose)
        pose_results = self.pose_evaluator.evaluate(gt_poses, pred_poses)
        
        # 4.5. 生成可视化 (如果指定了输出目录)
        if output_dir is not None:
            utils.log_evaluation_info("🎨 生成位姿可视化...", self.verbose)
            utils.ensure_dir(output_dir)
            
            # 重新计算对齐后的轨迹用于可视化
            from core.pose_evaluator import align_poses_with_scale
            pred_aligned, scale = align_poses_with_scale(gt_poses, pred_poses)
            
            # 计算ATE误差用于可视化
            gt_traj = gt_poses[:, :3, 3]
            ate_errors = np.linalg.norm(gt_traj - pred_aligned, axis=1)
            
            try:
                # 生成3D轨迹评估图
                vis_path1 = plot_trajectory_3d_evaluation(
                    gt_poses, pred_poses, pred_aligned, ate_errors,
                    output_dir, sequence_name
                )
                utils.log_evaluation_info(f"  ✅ 3D轨迹图: {vis_path1}", self.verbose)
                
                # 生成详细位姿分析图
                vis_path2 = plot_detailed_pose_analysis(
                    gt_poses, pred_poses, pred_aligned,
                    pose_results['pose_metrics']['gate'],
                    pose_results['pose_metrics']['late'],
                    output_dir, sequence_name
                )
                utils.log_evaluation_info(f"  ✅ 详细分析图: {vis_path2}", self.verbose)
                
                # 添加可视化路径到结果中
                pose_results['visualization'] = {
                    'trajectory_3d': vis_path1,
                    'detailed_analysis': vis_path2
                }
                
            except Exception as e:
                utils.log_evaluation_info(f"  ⚠️ 可视化生成失败: {e}", self.verbose)
        
        # 5. 添加输入信息
        pose_results['input_info'] = {
            'gt_path': gt_pose_path,
            'pred_path': pred_pose_path,
            'gt_format': gt_format,
            'pred_format': pred_format,
            'gt_unit': gt_unit,
            'pred_unit': pred_unit,
            'gt_validation': gt_validation,
            'pred_validation': pred_validation
        }
        
        utils.log_evaluation_info("✅ 位姿评估完成", self.verbose)
        return pose_results
    
    def evaluate_complete(self,
                         gt_depth_path: str,
                         pred_depth_path: str,
                         gt_pose_path: str,
                         pred_pose_path: str,
                         output_dir: str,
                         gt_depth_format: str = 'npz_file',
                         pred_depth_format: str = 'npy_dir',
                         gt_pose_format: str = 'npz_file',
                         pred_pose_format: str = 'npz_dir',
                         gt_depth_unit: str = 'm',
                         pred_depth_unit: str = 'm',
                         gt_pose_unit: str = 'm',
                         pred_pose_unit: str = 'm',
                         depth_mode: str = 'both',
                         sequence_name: str = "sequence",
                         **format_kwargs) -> Dict[str, Any]:
        """
        完整评估 (深度 + 位姿)
        
        Args:
            gt_depth_path: GT深度数据路径
            pred_depth_path: 预测深度数据路径
            gt_pose_path: GT位姿数据路径
            pred_pose_path: 预测位姿数据路径
            output_dir: 结果输出目录
            gt_depth_format: GT深度数据格式
            pred_depth_format: 预测深度数据格式
            gt_pose_format: GT位姿数据格式
            pred_pose_format: 预测位姿数据格式
            gt_depth_unit: GT深度单位
            pred_depth_unit: 预测深度单位
            gt_pose_unit: GT位姿位移单位
            pred_pose_unit: 预测位姿位移单位
            depth_mode: 深度评估模式 ('global'=全局中值缩放, 'segmented'=分段中值缩放, 'both'=两种方法)
            **format_kwargs: 格式特定参数
        
        Returns:
            完整评估结果
        """
        utils.log_evaluation_info("🚀 开始完整模型评估 (深度 + 位姿)", self.verbose)
        
        # 创建输出目录
        utils.ensure_dir(output_dir)
        
        # 1. 深度评估
        depth_results = self.evaluate_depth_only(
            gt_depth_path, pred_depth_path,
            gt_depth_format, pred_depth_format,
            gt_depth_unit, pred_depth_unit,
            depth_mode,
            **format_kwargs
        )
        
        # 2. 位姿评估  
        pose_results = self.evaluate_pose_only(
            gt_pose_path, pred_pose_path,
            gt_pose_format, pred_pose_format,
            gt_pose_unit, pred_pose_unit,
            output_dir, sequence_name,
            **format_kwargs
        )
        
        # 3. 汇总结果
        complete_results = {
            'depth_evaluation': depth_results,
            'pose_evaluation': pose_results,
            'evaluation_summary': {
                'evaluation_time': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'evaluation_type': 'complete',
                'input_paths': {
                    'gt_depth': gt_depth_path,
                    'pred_depth': pred_depth_path,
                    'gt_pose': gt_pose_path,
                    'pred_pose': pred_pose_path
                },
                'configuration': {
                    'depth_range_m': [self.depth_min, self.depth_max],
                    'pose_window_size': self.pose_window_size
                }
            }
        }
        
        # 4. 保存详细结果
        results_path = os.path.join(output_dir, 'complete_evaluation_results.json')
        utils.save_evaluation_results(complete_results, results_path)
        
        # 5. 生成简化报告
        self._generate_summary_report(complete_results, output_dir)
        
        utils.log_evaluation_info(f"📄 评估结果已保存到: {output_dir}", self.verbose)
        utils.log_evaluation_info("🎉 完整评估完成！", self.verbose)
        
        return complete_results
    
    def _generate_summary_report(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        生成简化的评估报告
        
        Args:
            results: 完整评估结果
            output_dir: 输出目录
        """
        report_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("统一评估框架 - 模型评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            summary = results['evaluation_summary']
            f.write(f"评估时间: {summary['evaluation_time']}\n")
            f.write(f"框架版本: {summary['framework_version']}\n")
            f.write(f"评估类型: {summary['evaluation_type']}\n\n")
            
            # 输入信息
            f.write("输入数据:\n")
            paths = summary['input_paths']
            f.write(f"  GT深度:   {paths['gt_depth']}\n")
            f.write(f"  预测深度: {paths['pred_depth']}\n")
            f.write(f"  GT位姿:   {paths['gt_pose']}\n")
            f.write(f"  预测位姿: {paths['pred_pose']}\n\n")
            
            # 深度评估结果
            depth_eval = results['depth_evaluation']
            f.write("🎯 深度评估结果:\n")
            f.write("-" * 40 + "\n")
            
            # 检查是否为双重评估结果
            if 'depth_metrics_global' in depth_eval and 'depth_metrics_segmented' in depth_eval:
                # 双重评估模式
                f.write("全局中值缩放结果:\n")
                global_metrics = depth_eval['depth_metrics_global']
                f.write(f"  abs_rel:  {global_metrics['abs_rel']['mean']:.4f} ± {global_metrics['abs_rel']['std']:.4f}\n")
                f.write(f"  sq_rel:   {global_metrics['sq_rel']['mean']*1000:.6f} ± {global_metrics['sq_rel']['std']*1000:.6f} mm\n")
                f.write(f"  rmse:     {global_metrics['rmse']['mean']*1000:.4f} ± {global_metrics['rmse']['std']*1000:.4f} mm\n")
                f.write(f"  rmse_log: {global_metrics['rmse_log']['mean']:.4f} ± {global_metrics['rmse_log']['std']:.4f}\n")
                f.write(f"  a1:       {global_metrics['a1']['mean']:.4f} ± {global_metrics['a1']['std']:.4f}\n")
                f.write(f"  a2:       {global_metrics['a2']['mean']:.4f} ± {global_metrics['a2']['std']:.4f}\n")
                f.write(f"  a3:       {global_metrics['a3']['mean']:.4f} ± {global_metrics['a3']['std']:.4f}\n\n")
                
                f.write("分段中值缩放结果 (16帧/段):\n")
                segmented_metrics = depth_eval['depth_metrics_segmented']
                f.write(f"  abs_rel:  {segmented_metrics['abs_rel']['mean']:.4f} ± {segmented_metrics['abs_rel']['std']:.4f}\n")
                f.write(f"  sq_rel:   {segmented_metrics['sq_rel']['mean']*1000:.6f} ± {segmented_metrics['sq_rel']['std']*1000:.6f} mm\n")
                f.write(f"  rmse:     {segmented_metrics['rmse']['mean']*1000:.4f} ± {segmented_metrics['rmse']['std']*1000:.4f} mm\n")
                f.write(f"  rmse_log: {segmented_metrics['rmse_log']['mean']:.4f} ± {segmented_metrics['rmse_log']['std']:.4f}\n")
                f.write(f"  a1:       {segmented_metrics['a1']['mean']:.4f} ± {segmented_metrics['a1']['std']:.4f}\n")
                f.write(f"  a2:       {segmented_metrics['a2']['mean']:.4f} ± {segmented_metrics['a2']['std']:.4f}\n")
                f.write(f"  a3:       {segmented_metrics['a3']['mean']:.4f} ± {segmented_metrics['a3']['std']:.4f}\n\n")
            else:
                # 单一评估模式
                depth_metrics = depth_eval['depth_metrics']
                f.write(f"  abs_rel:  {depth_metrics['abs_rel']['mean']:.4f} ± {depth_metrics['abs_rel']['std']:.4f}\n")
                f.write(f"  sq_rel:   {depth_metrics['sq_rel']['mean']*1000:.6f} ± {depth_metrics['sq_rel']['std']*1000:.6f} mm\n")
                f.write(f"  rmse:     {depth_metrics['rmse']['mean']*1000:.4f} ± {depth_metrics['rmse']['std']*1000:.4f} mm\n")
                f.write(f"  rmse_log: {depth_metrics['rmse_log']['mean']:.4f} ± {depth_metrics['rmse_log']['std']:.4f}\n")
                f.write(f"  a1:       {depth_metrics['a1']['mean']:.4f} ± {depth_metrics['a1']['std']:.4f}\n")
                f.write(f"  a2:       {depth_metrics['a2']['mean']:.4f} ± {depth_metrics['a2']['std']:.4f}\n")
                f.write(f"  a3:       {depth_metrics['a3']['mean']:.4f} ± {depth_metrics['a3']['std']:.4f}\n\n")
            
            # 位姿评估结果
            pose_metrics = results['pose_evaluation']['pose_metrics']
            gate = pose_metrics['gate']
            late = pose_metrics['late']
            
            f.write("🎯 位姿评估结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  G-ATE RMSE: {gate['gate_rmse']:.4f} m\n")
            f.write(f"  G-ATE Mean: {gate['gate_mean']:.4f} m\n")
            f.write(f"  G-ATE Std:  {gate['gate_std']:.4f} m\n")
            f.write(f"  尺度因子:   {gate['alignment_info']['scale_factor']:.6f}\n\n")
            
            f.write(f"  L-ATE RMSE: {late['late_rmse_mean']:.4f} ± {late['late_rmse_std']:.4f} m\n")
            f.write(f"  窗口大小:   {late['evaluation_info']['window_size']}\n")
            f.write(f"  窗口数量:   {late['evaluation_info']['total_windows']}/{late['evaluation_info']['expected_windows']}\n")
            f.write(f"  成功率:     {late['evaluation_info']['success_rate']:.1%}\n\n")
            
            # 性能评级
            # 选择合适的深度指标进行评级
            if 'depth_metrics_segmented' in results['depth_evaluation']:
                a1_score = results['depth_evaluation']['depth_metrics_segmented']['a1']['mean']
            else:
                a1_score = results['depth_evaluation']['depth_metrics']['a1']['mean']
            gate_rmse = gate['gate_rmse']
            late_rmse = late['late_rmse_mean']
            
            f.write("🏆 性能评级:\n")
            f.write("-" * 40 + "\n")
            
            # 深度评级
            if a1_score > 0.8:
                depth_grade = "S (优秀)"
            elif a1_score > 0.6:
                depth_grade = "A (良好)"
            else:
                depth_grade = "B (有待改进)"
            f.write(f"  深度评级: {depth_grade} (a1={a1_score:.3f})\n")
            
            # 位姿评级
            if gate_rmse < 0.01 and late_rmse < 0.01:
                pose_grade = "S (优秀)"
            elif gate_rmse < 0.05 and late_rmse < 0.05:
                pose_grade = "A (良好)"
            else:
                pose_grade = "B (有待改进)"
            f.write(f"  位姿评级: {pose_grade} (G-ATE={gate_rmse:.3f}m, L-ATE={late_rmse:.3f}m)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        utils.log_evaluation_info(f"📋 评估报告已保存: {report_path}", self.verbose)
    
    def print_complete_results(self, results: Dict[str, Any]) -> None:
        """
        打印完整评估结果摘要
        
        Args:
            results: 完整评估结果
        """
        print("\n" + "="*80)
        print("🎉 统一评估框架 - 完整评估结果")
        print("="*80)
        
        # 打印深度结果
        if 'depth_metrics_global' in results['depth_evaluation'] and 'depth_metrics_segmented' in results['depth_evaluation']:
            # 双重评估模式 - 分别打印全局和分段结果
            print("\n📊 全局中值缩放深度评估结果:")
            global_result = {
                'depth_metrics': results['depth_evaluation']['depth_metrics_global'],
                'evaluation_info': results['depth_evaluation']['evaluation_info']['global_info'],
                'processing_stats': results['depth_evaluation']['processing_stats']['global_stats']
            }
            self.depth_evaluator.print_results(global_result)
            
            print("\n📊 分段中值缩放深度评估结果 (16帧/段):")
            segmented_result = {
                'depth_metrics': results['depth_evaluation']['depth_metrics_segmented'], 
                'evaluation_info': results['depth_evaluation']['evaluation_info']['segmented_info'],
                'processing_stats': results['depth_evaluation']['processing_stats']['segmented_stats']
            }
            self.depth_evaluator.print_results(segmented_result)
        else:
            # 单一评估模式
            self.depth_evaluator.print_results(results['depth_evaluation'])
        
        # 打印位姿结果
        self.pose_evaluator.print_results(results['pose_evaluation'])
        
        # 打印总结
        print("\n" + "="*80)
        print("🏆 总体评估总结")
        print("="*80)
        
        # 获取深度a1指标（如果是双重评估，使用分段评估结果）
        if 'depth_metrics_global' in results['depth_evaluation'] and 'depth_metrics_segmented' in results['depth_evaluation']:
            depth_a1 = results['depth_evaluation']['depth_metrics_segmented']['a1']['mean']
        else:
            depth_a1 = results['depth_evaluation']['depth_metrics']['a1']['mean']
        gate_rmse = results['pose_evaluation']['pose_metrics']['gate']['gate_rmse']
        late_rmse = results['pose_evaluation']['pose_metrics']['late']['late_rmse_mean']
        
        print(f"深度性能: a1={depth_a1:.3f}")
        print(f"位姿性能: G-ATE={gate_rmse:.4f}m, L-ATE={late_rmse:.4f}m")
        print(f"评估框架: 统一标准，格式兼容，结果可靠")
        print("="*80)
