"""
位姿评估核心模块
严格按照统一评估标准实现位姿评估流程
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from core.utils import (
    compute_statistics, log_evaluation_info, validate_input_shapes
)
from core.visualization import plot_trajectory_3d_evaluation, plot_detailed_pose_analysis


def align_poses_with_scale(gt_poses: np.ndarray, pred_poses: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    位姿尺度对齐 - 严格按照评估标准
    
    流程:
    1. 起始帧对齐 (pred[0] = gt[0])
    2. 最小二乘法估计尺度因子
    3. 应用尺度变换并保证起点仍对齐
    
    Args:
        gt_poses: (N, 4, 4) GT位姿矩阵，cam2world格式，位移单位m
        pred_poses: (N, 4, 4) 预测位姿矩阵，cam2world格式，位移单位m
    
    Returns:
        对齐后的预测轨迹 (N, 3), 尺度因子
    """
    # 提取位移轨迹
    gt_traj = gt_poses[:, :3, 3]      # (N, 3)
    pred_traj = pred_poses[:, :3, 3]  # (N, 3)
    
    # 步骤1: 中心化（以起点为原点）
    gt_centered = gt_traj - gt_traj[0]      # 起点移至原点
    pred_centered = pred_traj - pred_traj[0]
    
    # 步骤2: 最小二乘法计算尺度因子
    # scale = argmin_s ||gt_centered - s * pred_centered||^2
    numerator = np.sum(gt_centered * pred_centered)
    denominator = np.sum(pred_centered ** 2)
    
    if denominator == 0:
        # 预测轨迹为静态点
        scale = 1.0
        log_evaluation_info("警告: 预测轨迹为静态点，设置尺度因子为1.0")
    else:
        scale = numerator / denominator
    
    # 步骤3: 应用尺度变换
    pred_scaled = pred_centered * scale
    
    # 步骤4: 平移到GT起点（保证起点对齐）
    pred_aligned = pred_scaled + gt_traj[0]
    
    # 验证起点对齐
    start_error = np.linalg.norm(gt_traj[0] - pred_aligned[0])
    if start_error > 1e-10:
        log_evaluation_info(f"警告: 起点对齐误差 {start_error:.2e} m")
    
    return pred_aligned, scale


def compute_ate(gt_traj: np.ndarray, pred_traj_aligned: np.ndarray) -> Dict[str, float]:
    """
    计算绝对轨迹误差 (ATE)
    
    Args:
        gt_traj: (N, 3) GT轨迹
        pred_traj_aligned: (N, 3) 对齐后的预测轨迹
    
    Returns:
        ATE统计指标
    """
    # 计算点对点欧几里得距离
    errors = np.linalg.norm(gt_traj - pred_traj_aligned, axis=1)
    
    # 计算统计指标
    ate_stats = compute_statistics(errors)
    
    return {
        'rmse': ate_stats['rmse'],
        'mean': ate_stats['mean'],
        'std': ate_stats['std'],
        'median': ate_stats['median'],
        'max': ate_stats['max'],
        'min': ate_stats['min'],
        'errors': errors
    }


class PoseEvaluator:
    """
    位姿评估器 - 统一位姿评估标准
    
    支持两种评估模式:
    1. G-ATE (Global ATE): 全局起始帧对齐，计算整个序列的ATE
    2. L-ATE (Local ATE): 不重叠16帧窗口，每个窗口独立对齐并计算ATE
    """
    
    def __init__(self, window_size: int = 16, verbose: bool = True):
        """
        初始化位姿评估器
        
        Args:
            window_size: L-ATE滑动窗口大小（不重叠）
            verbose: 是否打印详细信息
        """
        self.window_size = window_size
        self.verbose = verbose
        
        log_evaluation_info(f"位姿评估器初始化 - L-ATE窗口大小: {window_size} (不重叠)", verbose)
    
    def evaluate_global_ate(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> Dict[str, Any]:
        """
        G-ATE评估: 全局起始帧对齐
        
        Args:
            gt_poses: (N, 4, 4) GT位姿序列
            pred_poses: (N, 4, 4) 预测位姿序列
        
        Returns:
            G-ATE评估结果
        """
        log_evaluation_info("🎯 开始G-ATE评估 (全局对齐)", self.verbose)
        
        # 全局对齐
        pred_traj_aligned, scale = align_poses_with_scale(gt_poses, pred_poses)
        gt_traj = gt_poses[:, :3, 3]
        
        # 计算ATE
        ate_result = compute_ate(gt_traj, pred_traj_aligned)
        
        # 构造结果
        result = {
            'gate_rmse': ate_result['rmse'],
            'gate_mean': ate_result['mean'],
            'gate_std': ate_result['std'],
            'alignment_info': {
                'scale_factor': scale,
                'start_point_aligned': True,
                'method': 'least_squares_scaling'
            },
            'detailed_stats': ate_result
        }
        
        log_evaluation_info(f"G-ATE RMSE: {ate_result['rmse']*1000:.3f} mm", self.verbose)
        return result
    
    def evaluate_local_ate(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> Dict[str, Any]:
        """
        L-ATE评估: 不重叠窗口局部对齐
        
        流程:
        1. 将序列分割为不重叠的window_size帧窗口 (0-15, 16-31, 32-47...)
        2. 每个窗口独立执行G-ATE相同流程（起始帧对齐+尺度估计）
        3. 计算每个窗口的ATE RMSE
        4. 汇总所有窗口的RMSE统计信息 (mean, std)
        
        Args:
            gt_poses: (N, 4, 4) GT位姿序列
            pred_poses: (N, 4, 4) 预测位姿序列
        
        Returns:
            L-ATE评估结果
        """
        log_evaluation_info(f"🎯 开始L-ATE评估 (不重叠{self.window_size}帧窗口)", self.verbose)
        
        num_frames = len(gt_poses)
        num_windows = num_frames // self.window_size
        
        if num_windows == 0:
            raise ValueError(f"序列长度 {num_frames} 不足一个窗口 ({self.window_size})")
        
        log_evaluation_info(f"总帧数: {num_frames}, 窗口数: {num_windows}", self.verbose)
        
        window_rmses = []
        window_details = []
        
        # 不重叠窗口遍历
        for i in tqdm(range(num_windows), desc="L-ATE窗口评估", disable=not self.verbose):
            start_idx = i * self.window_size
            end_idx = start_idx + self.window_size
            
            # 提取窗口数据
            gt_window = gt_poses[start_idx:end_idx]
            pred_window = pred_poses[start_idx:end_idx]
            
            try:
                # 窗口内对齐 (每个窗口独立执行G-ATE流程)
                pred_traj_aligned, scale = align_poses_with_scale(gt_window, pred_window)
                gt_traj = gt_window[:, :3, 3]
                
                # 计算窗口内ATE
                ate_result = compute_ate(gt_traj, pred_traj_aligned)
                window_rmse = ate_result['rmse']
                
                window_rmses.append(window_rmse)
                window_details.append({
                    'window_id': i,
                    'frame_range': (start_idx, end_idx-1),
                    'rmse': window_rmse,
                    'scale_factor': scale,
                    'detailed_stats': ate_result
                })
                
                if self.verbose and i < 3:  # 只打印前几个窗口的详情
                    log_evaluation_info(f"窗口{i} [{start_idx}-{end_idx-1}]: RMSE={window_rmse*1000:.1f}mm, scale={scale:.4f}", self.verbose)
                
            except Exception as e:
                log_evaluation_info(f"窗口{i} [{start_idx}-{end_idx-1}] 评估失败: {e}", self.verbose)
                continue
        
        if len(window_rmses) == 0:
            raise ValueError("没有成功评估任何窗口")
        
        # 汇总统计信息
        window_rmses = np.array(window_rmses)
        rmse_stats = compute_statistics(window_rmses)
        
        # 构造结果
        result = {
            'late_rmse_mean': rmse_stats['mean'],      # 所有窗口RMSE的平均值
            'late_rmse_std': rmse_stats['std'],        # 所有窗口RMSE的标准差
            'late_rmse_median': rmse_stats['median'],
            'late_rmse_max': rmse_stats['max'],
            'late_rmse_min': rmse_stats['min'],
            'evaluation_info': {
                'window_size': self.window_size,
                'total_windows': len(window_rmses),
                'expected_windows': num_windows,
                'success_rate': len(window_rmses) / num_windows
            },
            'per_window_details': window_details,
            'all_window_rmses': window_rmses.tolist()
        }
        
        log_evaluation_info(f"L-ATE结果: RMSE={rmse_stats['mean']*1000:.3f}±{rmse_stats['std']*1000:.3f} mm ({len(window_rmses)}/{num_windows}窗口)", self.verbose)
        return result
    
    def evaluate(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> Dict[str, Any]:
        """
        完整位姿评估 (包含G-ATE和L-ATE)
        
        Args:
            gt_poses: (N, 4, 4) GT位姿序列，单位m
            pred_poses: (N, 4, 4) 预测位姿序列，单位m
        
        Returns:
            完整位姿评估结果
        """
        log_evaluation_info("🚀 开始位姿评估", self.verbose)
        
        # 验证输入
        validate_input_shapes(gt_poses, pred_poses, 'pose')
        num_frames = len(gt_poses)
        
        log_evaluation_info(f"总帧数: {num_frames}", self.verbose)
        log_evaluation_info(f"GT位姿形状: {gt_poses.shape}", self.verbose)
        log_evaluation_info(f"预测位姿形状: {pred_poses.shape}", self.verbose)
        
        # G-ATE评估
        gate_result = self.evaluate_global_ate(gt_poses, pred_poses)
        
        # L-ATE评估
        late_result = self.evaluate_local_ate(gt_poses, pred_poses)
        
        # 汇总结果
        results = {
            'pose_metrics': {
                'gate': gate_result,
                'late': late_result
            },
            'evaluation_info': {
                'total_frames': num_frames,
                'evaluation_method': 'G-ATE + L-ATE',
                'alignment_method': 'start_point_align + least_squares_scaling',
                'window_strategy': f'non_overlapping_{self.window_size}_frames'
            }
        }
        
        log_evaluation_info("✅ 位姿评估完成", self.verbose)
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        打印位姿评估结果摘要
        
        Args:
            results: 位姿评估结果
        """
        print("\n" + "="*70)
        print("🎯 位姿评估结果摘要")
        print("="*70)
        
        # 基本信息
        info = results['evaluation_info']
        print(f"评估帧数: {info['total_frames']}")
        print(f"评估方法: {info['evaluation_method']}")
        
        # G-ATE结果
        gate = results['pose_metrics']['gate']
        print(f"\n📊 G-ATE (全局):")
        print(f"  RMSE: {gate['gate_rmse']*1000:.3f} mm")
        print(f"  Mean: {gate['gate_mean']*1000:.3f} mm")
        print(f"  Std:  {gate['gate_std']*1000:.3f} mm")
        print(f"  尺度因子: {gate['alignment_info']['scale_factor']:.6f}")
        
        # L-ATE结果
        late = results['pose_metrics']['late']
        print(f"\n📊 L-ATE (局部{late['evaluation_info']['window_size']}帧窗口):")
        print(f"  RMSE Mean: {late['late_rmse_mean']*1000:.3f} mm")
        print(f"  RMSE Std:  {late['late_rmse_std']*1000:.3f} mm")
        print(f"  窗口数: {late['evaluation_info']['total_windows']}/{late['evaluation_info']['expected_windows']}")
        print(f"  成功率: {late['evaluation_info']['success_rate']:.1%}")
        
        # 性能总结
        gate_rmse = gate['gate_rmse']
        late_rmse = late['late_rmse_mean']
        
        print(f"\n🏆 性能总结:")
        print(f"  G-ATE: {gate_rmse*1000:.3f} mm")
        print(f"  L-ATE: {late_rmse*1000:.3f} ± {late['late_rmse_std']*1000:.3f} mm")
        
        if gate_rmse < 0.01 and late_rmse < 0.01:
            grade = "优秀 (S级)"
        elif gate_rmse < 0.05 and late_rmse < 0.05:
            grade = "良好 (A级)"
        else:
            grade = "有待改进 (B级)"
        
        print(f"  总体评级: {grade}")
        print("="*70)
