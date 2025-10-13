"""
统一评估框架可视化模块
提供深度和位姿评估的可视化功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional


def compute_unified_axis_limits(gt_traj: np.ndarray, pred_aligned: np.ndarray = None, 
                              margin_factor: float = 0.15) -> Dict[str, tuple]:
    """
    计算统一的坐标轴范围，确保多张图具有相同的坐标系
    主要基于GT轨迹计算范围，确保不同图之间的一致性
    
    Args:
        gt_traj: (N, 3) GT轨迹
        pred_aligned: (N, 3) 对齐后的预测轨迹（可选，用于确保预测轨迹也在范围内）
        margin_factor: 边距因子，用于在数据范围外留一些空间
    
    Returns:
        dict: 包含xlim, ylim, zlim的字典
    """
    # 主要基于GT轨迹计算范围，确保一致性（保持原始米单位）
    reference_points = gt_traj
    
    # 如果提供了对齐后的预测轨迹，确保它也在范围内
    if pred_aligned is not None:
        # 合并数据但主要权重给GT
        all_points = np.vstack([gt_traj, pred_aligned])
        reference_points = all_points
    
    # 计算每个轴的最小值和最大值
    min_vals = np.min(reference_points, axis=0)
    max_vals = np.max(reference_points, axis=0)
    
    # 计算范围并添加边距
    ranges = max_vals - min_vals
    margins = ranges * margin_factor
    
    # 确保有最小范围，避免轴太窄
    min_range = 0.1  # 最小范围 0.1m
    for i in range(3):
        if ranges[i] < min_range:
            ranges[i] = min_range
            margins[i] = min_range * margin_factor
    
    # 计算最终范围（米单位）
    limits = {
        'xlim': (min_vals[0] - margins[0], max_vals[0] + margins[0]),
        'ylim': (min_vals[1] - margins[1], max_vals[1] + margins[1]),
        'zlim': (min_vals[2] - margins[2], max_vals[2] + margins[2])
    }
    
    return limits


def apply_unified_3d_axis_settings(ax, limits: Dict[str, tuple], 
                                  elev: float = 20., azim: float = -150):
    """
    应用统一的3D坐标轴设置
    
    Args:
        ax: matplotlib 3D轴对象
        limits: 坐标轴范围字典
        elev: 视角仰角
        azim: 视角方位角
    """
    ax.set_xlim(limits['xlim'])
    ax.set_ylim(limits['ylim'])
    ax.set_zlim(limits['zlim'])
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    
    # 设置轴标签
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')


def plot_trajectory_3d_evaluation(gt_poses: np.ndarray, pred_poses: np.ndarray, 
                                 pred_aligned: np.ndarray, ate_errors: np.ndarray,
                                 output_dir: str, sequence_name: str = "sequence") -> str:
    """
    生成3D轨迹评估可视化 (与原有SCARED评估相同风格)
    
    Args:
        gt_poses: (N, 4, 4) GT位姿矩阵
        pred_poses: (N, 4, 4) 原始预测位姿矩阵  
        pred_aligned: (N, 3) 对齐后的预测轨迹
        ate_errors: (N,) 每帧ATE误差
        output_dir: 输出目录
        sequence_name: 序列名称
    
    Returns:
        保存的图片路径
    """
    # 提取轨迹
    gt_traj = gt_poses[:, :3, 3]
    pred_traj_original = pred_poses[:, :3, 3]
    
    # 计算统一的坐标轴范围（基于GT和对齐后的预测轨迹）
    axis_limits = compute_unified_axis_limits(gt_traj, pred_aligned)
    
    # 创建3个子图的可视化
    fig = plt.figure(figsize=(18, 6))
    
    # 子图1: 原始轨迹对比
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'b-', linewidth=2, label='GT Trajectory')
    ax1.plot(pred_traj_original[:, 0], pred_traj_original[:, 1], pred_traj_original[:, 2], 
             'r--', linewidth=2, label='Predicted (Original)')
    ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], c='b', marker='o', s=100, label='GT Start')
    ax1.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], c='b', marker='s', s=100, label='GT End')
    ax1.legend()
    ax1.set_title('Original Trajectories')
    apply_unified_3d_axis_settings(ax1, axis_limits)
    
    # 子图2: 对齐后轨迹对比
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], 'b-', linewidth=2, label='GT Trajectory')
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2], 
             'g-', linewidth=2, label='Predicted (Aligned)')
    ax2.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], c='b', marker='o', s=100, label='Start')
    ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], c='b', marker='s', s=100, label='End')
    ax2.scatter(pred_aligned[0, 0], pred_aligned[0, 1], pred_aligned[0, 2], c='g', marker='o', s=100)
    ax2.scatter(pred_aligned[-1, 0], pred_aligned[-1, 1], pred_aligned[-1, 2], c='g', marker='s', s=100)
    ax2.legend()
    ax2.set_title('Scale-Aligned Trajectories')
    apply_unified_3d_axis_settings(ax2, axis_limits)
    
    # 子图3: 每帧轨迹误差
    ax3 = fig.add_subplot(133)
    frames = range(len(ate_errors))
    ax3.plot(frames, ate_errors, 'r-', alpha=0.8)
    ax3.fill_between(frames, ate_errors, color='red', alpha=0.3)
    
    mean_error = np.mean(ate_errors)
    median_error = np.median(ate_errors)
    
    ax3.axhline(mean_error, color='g', linestyle='--', label=f'Mean Error: {mean_error:.3f}')
    ax3.axhline(median_error, color='orange', linestyle='--', label=f'Median Error: {median_error:.3f}')
    
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position Error')
    ax3.set_title('Trajectory Error per Frame')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'trajectory_evaluation_{sequence_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_detailed_pose_analysis(gt_poses: np.ndarray, pred_poses: np.ndarray,
                               pred_aligned: np.ndarray, gate_result: Dict[str, Any],
                               late_result: Dict[str, Any], output_dir: str, 
                               sequence_name: str = "sequence") -> str:
    """
    生成详细的位姿分析可视化
    
    Args:
        gt_poses: GT位姿
        pred_poses: 预测位姿
        pred_aligned: 对齐后轨迹
        gate_result: G-ATE结果
        late_result: L-ATE结果
        output_dir: 输出目录
        sequence_name: 序列名称
    
    Returns:
        保存的图片路径
    """
    gt_traj = gt_poses[:, :3, 3]
    pred_traj = pred_poses[:, :3, 3]
    
    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. XY平面轨迹
    ax1.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=2, label='GT')
    ax1.plot(pred_aligned[:, 0], pred_aligned[:, 1], 'g-', linewidth=2, label='Predicted (Aligned)')
    ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], c='blue', s=100, marker='o', label='Start')
    ax1.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('XY Plane Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # 2. XZ平面轨迹
    ax2.plot(gt_traj[:, 0], gt_traj[:, 2], 'b-', linewidth=2, label='GT')
    ax2.plot(pred_aligned[:, 0], pred_aligned[:, 2], 'g-', linewidth=2, label='Predicted (Aligned)')
    ax2.scatter(gt_traj[0, 0], gt_traj[0, 2], c='blue', s=100, marker='o')
    ax2.scatter(gt_traj[-1, 0], gt_traj[-1, 2], c='red', s=100, marker='s')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('XZ Plane Trajectory')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. G-ATE误差时间线
    if 'errors' in gate_result['detailed_stats']:
        gate_errors = gate_result['detailed_stats']['errors']
        frames = np.arange(len(gate_errors))
        ax3.plot(frames, gate_errors, 'r-', linewidth=1, alpha=0.8)
        ax3.axhline(y=gate_result['gate_mean'], color='g', linestyle='--', 
                   label=f'Mean: {gate_result["gate_mean"]:.3f}')
        ax3.axhline(y=gate_result['gate_rmse'], color='b', linestyle='--',
                   label=f'RMSE: {gate_result["gate_rmse"]:.3f}')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('G-ATE Error')
        ax3.set_title('Global ATE Error Timeline')
        ax3.legend()
        ax3.grid(True)
    
    # 4. L-ATE窗口误差分布
    if 'all_window_rmses' in late_result:
        window_rmses = late_result['all_window_rmses']
        windows = np.arange(len(window_rmses))
        ax4.plot(windows, window_rmses, 'b-', marker='o', linewidth=1, markersize=4)
        ax4.axhline(y=late_result['late_rmse_mean'], color='g', linestyle='--',
                   label=f'Mean: {late_result["late_rmse_mean"]:.3f}')
        ax4.fill_between(windows, 
                        late_result['late_rmse_mean'] - late_result['late_rmse_std'],
                        late_result['late_rmse_mean'] + late_result['late_rmse_std'],
                        color='green', alpha=0.2, label=f'±1σ: ±{late_result["late_rmse_std"]:.3f}')
        ax4.set_xlabel('Window Index')
        ax4.set_ylabel('L-ATE RMSE')
        ax4.set_title(f'Local ATE (Window Size: {late_result["evaluation_info"]["window_size"]})')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, f'pose_analysis_{sequence_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_depth_comparison(gt_depths: np.ndarray, pred_depths: np.ndarray,
                         frame_indices: list, output_dir: str,
                         sequence_name: str = "sequence") -> str:
    """
    生成深度图对比可视化
    
    Args:
        gt_depths: (N, H, W) GT深度
        pred_depths: (N, H, W) 预测深度
        frame_indices: 要可视化的帧索引列表
        output_dir: 输出目录
        sequence_name: 序列名称
    
    Returns:
        保存的图片路径
    """
    num_frames = len(frame_indices)
    fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 12))
    
    if num_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        # GT深度
        gt_depth = gt_depths[frame_idx]
        im1 = axes[0, i].imshow(gt_depth, cmap='plasma', vmin=0, vmax=np.percentile(gt_depth, 95))
        axes[0, i].set_title(f'GT Depth\nFrame {frame_idx}')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        
        # 预测深度
        pred_depth = pred_depths[frame_idx]
        im2 = axes[1, i].imshow(pred_depth, cmap='plasma', vmin=0, vmax=np.percentile(gt_depth, 95))
        axes[1, i].set_title(f'Predicted Depth\nFrame {frame_idx}')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
        
        # 误差图
        error_map = np.abs(gt_depth - pred_depth)
        im3 = axes[2, i].imshow(error_map, cmap='hot', vmin=0, vmax=np.percentile(error_map, 95))
        axes[2, i].set_title(f'Error Map\nFrame {frame_idx}')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(output_dir, f'depth_comparison_{sequence_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path



