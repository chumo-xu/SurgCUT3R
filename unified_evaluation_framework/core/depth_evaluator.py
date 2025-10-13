"""
深度评估核心模块
严格按照统一评估标准实现深度评估流程
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from core.utils import (
    resize_depth, create_depth_mask, median_scale_alignment, 
    compute_statistics, log_evaluation_info, validate_input_shapes
)


def compute_depth_errors(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    计算深度估计误差指标（与EndoDAC标准一致）
    
    Args:
        gt: GT深度值 (有效像素)
        pred: 预测深度值 (有效像素)
    
    Returns:
        误差指标字典
    """
    # 计算阈值指标
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean() 
    a3 = (thresh < 1.25 ** 3).mean()

    # 计算RMSE指标
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

    # 计算相对误差指标
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel), 
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3)
    }


class DepthEvaluator:
    """
    深度评估器 - 统一深度评估标准
    
    评估流程:
    1. 调整到GT尺寸
    2. 创建有效掩码 (1e-3到150m)
    3. 中值缩放对齐 
    4. 深度范围截断
    5. 计算误差指标
    """
    
    def __init__(self, min_depth: float = 1e-3, max_depth: float = 150.0, 
                 min_valid_pixels: int = 1000, verbose: bool = True):
        """
        初始化深度评估器
        
        Args:
            min_depth: 最小有效深度 (m)
            max_depth: 最大有效深度 (m)
            min_valid_pixels: 最小有效像素数
            verbose: 是否打印详细信息
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_valid_pixels = min_valid_pixels
        self.verbose = verbose
        
        log_evaluation_info(f"深度评估器初始化 - 有效深度范围: {min_depth:.1e}~{max_depth:.1f}m", verbose)
    
    def evaluate_frame(self, gt_depth: np.ndarray, pred_depth: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        评估单帧深度
        
        Args:
            gt_depth: (H, W) GT深度图，单位m
            pred_depth: (H', W') 预测深度图，单位m
        
        Returns:
            误差指标, 处理信息
        """
        processing_info = {}
        
        # 步骤1: 调整到GT尺寸
        target_height, target_width = gt_depth.shape
        if pred_depth.shape != gt_depth.shape:
            pred_depth_resized = resize_depth(pred_depth, (target_height, target_width))
            processing_info['resized'] = True
            processing_info['original_shape'] = pred_depth.shape
            processing_info['target_shape'] = (target_height, target_width)
        else:
            pred_depth_resized = pred_depth.copy()
            processing_info['resized'] = False
        
        # 步骤2: 创建有效掩码 (1e-3到150m范围)
        mask = create_depth_mask(gt_depth, pred_depth_resized, self.min_depth, self.max_depth)
        valid_pixels = np.sum(mask)
        processing_info['valid_pixels'] = int(valid_pixels)
        
        if valid_pixels < self.min_valid_pixels:
            raise ValueError(f"有效像素数太少: {valid_pixels} < {self.min_valid_pixels}")
        
        # 步骤3: 中值缩放对齐
        try:
            pred_depth_scaled, scale_ratio = median_scale_alignment(gt_depth, pred_depth_resized, mask)
            processing_info['scale_ratio'] = float(scale_ratio)
        except ValueError as e:
            raise ValueError(f"缩放对齐失败: {e}")
        
        # 步骤4: 深度范围截断 (限制预测深度的极值)
        pred_depth_clipped = np.clip(pred_depth_scaled, self.min_depth, self.max_depth)
        
        # 步骤5: 重新创建最终掩码，确保预测和真实深度都在合理范围内
        final_mask = create_depth_mask(gt_depth, pred_depth_clipped, self.min_depth, self.max_depth)
        final_valid_pixels = np.sum(final_mask)
        processing_info['final_valid_pixels'] = int(final_valid_pixels)
        
        if final_valid_pixels < self.min_valid_pixels // 2:  # 容忍最终有效像素减少
            raise ValueError(f"最终有效像素数太少: {final_valid_pixels}")
        
        # 应用最终掩码
        gt_final = gt_depth[final_mask]
        pred_final = pred_depth_clipped[final_mask]
        
        # 计算误差指标
        errors = compute_depth_errors(gt_final, pred_final)
        
        return errors, processing_info
    
    def evaluate_segment(self, gt_depths: np.ndarray, pred_depths: np.ndarray, segment_id: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        评估单个16帧分段的深度
        
        Args:
            gt_depths: (N, H, W) GT深度分段，单位m (N <= 16)
            pred_depths: (N, H', W') 预测深度分段，单位m
            segment_id: 分段ID用于日志
        
        Returns:
            分段平均误差指标, 处理信息
        """
        num_frames = len(gt_depths)
        log_evaluation_info(f"评估分段{segment_id} ({num_frames}帧)", self.verbose)
        
        # 收集所有有效像素进行统一中值缩放
        all_gt_valid = []
        all_pred_valid = []
        frame_masks = []
        
        target_height, target_width = gt_depths[0].shape
        
        # 第一步：收集所有有效像素
        for i in range(num_frames):
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]
            
            # 调整到GT尺寸
            if pred_depth.shape != gt_depth.shape:
                pred_depth = resize_depth(pred_depth, (target_height, target_width))
            
            # 创建有效掩码
            mask = create_depth_mask(gt_depth, pred_depth, self.min_depth, self.max_depth)
            
            if np.sum(mask) > 0:
                all_gt_valid.extend(gt_depth[mask])
                all_pred_valid.extend(pred_depth[mask])
                frame_masks.append(mask)
            else:
                frame_masks.append(None)
        
        if len(all_gt_valid) == 0:
            raise ValueError(f"分段{segment_id}没有有效像素")
        
        # 第二步：计算分段统一的中值缩放因子
        all_gt_valid = np.array(all_gt_valid)
        all_pred_valid = np.array(all_pred_valid)
        
        try:
            scale_ratio = np.median(all_gt_valid) / np.median(all_pred_valid)
        except:
            raise ValueError(f"分段{segment_id}中值缩放计算失败")
        
        # 第三步：对每帧应用统一缩放并计算指标
        segment_errors = []
        
        for i in range(num_frames):
            if frame_masks[i] is None:
                continue
                
            gt_depth = gt_depths[i]
            pred_depth = pred_depths[i]
            
            # 调整尺寸
            if pred_depth.shape != gt_depth.shape:
                pred_depth = resize_depth(pred_depth, (target_height, target_width))
            
            # 应用统一缩放
            pred_depth_scaled = pred_depth * scale_ratio
            
            # 深度截断
            pred_depth_clipped = np.clip(pred_depth_scaled, self.min_depth, self.max_depth)
            
            # 最终掩码
            final_mask = create_depth_mask(gt_depth, pred_depth_clipped, self.min_depth, self.max_depth)
            
            if np.sum(final_mask) > 0:
                gt_final = gt_depth[final_mask]
                pred_final = pred_depth_clipped[final_mask]
                
                # 计算单帧误差
                frame_error = compute_depth_errors(gt_final, pred_final)
                segment_errors.append(frame_error)
        
        if len(segment_errors) == 0:
            raise ValueError(f"分段{segment_id}没有成功评估任何帧")
        
        # 第四步：计算分段平均指标
        metrics_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        segment_metrics = {}
        
        for name in metrics_names:
            values = [err[name] for err in segment_errors]
            segment_metrics[name] = float(np.mean(values))
        
        processing_info = {
            'segment_id': segment_id,
            'num_frames': num_frames,
            'valid_frames': len(segment_errors),
            'scale_ratio': float(scale_ratio),
            'total_valid_pixels': len(all_gt_valid)
        }
        
        return segment_metrics, processing_info
    
    def evaluate_segmented(self, gt_depths: np.ndarray, pred_depths: np.ndarray, segment_size: int = 16) -> Dict[str, Any]:
        """
        分段评估深度序列（每段独立中值缩放）
        
        Args:
            gt_depths: (N, H, W) GT深度序列，单位m
            pred_depths: (N, H', W') 预测深度序列，单位m
            segment_size: 分段大小，默认16帧
        
        Returns:
            分段评估结果
        """
        log_evaluation_info(f"🎯 开始分段深度评估 (每{segment_size}帧一段)", self.verbose)
        
        # 验证输入
        validate_input_shapes(gt_depths, pred_depths, 'depth')
        num_frames = len(gt_depths)
        num_segments = (num_frames + segment_size - 1) // segment_size  # 向上取整
        
        log_evaluation_info(f"总帧数: {num_frames}, 分段数: {num_segments}", self.verbose)
        
        # 逐段评估
        segment_metrics = []
        segment_info = []
        failed_segments = []
        
        for seg_id in tqdm(range(num_segments), desc="评估分段", disable=not self.verbose):
            start_idx = seg_id * segment_size
            end_idx = min(start_idx + segment_size, num_frames)
            
            gt_segment = gt_depths[start_idx:end_idx]
            pred_segment = pred_depths[start_idx:end_idx]
            
            try:
                metrics, info = self.evaluate_segment(gt_segment, pred_segment, seg_id)
                segment_metrics.append(metrics)
                segment_info.append(info)
                
            except ValueError as e:
                if self.verbose:
                    log_evaluation_info(f"分段{seg_id} [{start_idx}-{end_idx-1}] 评估失败: {e}", self.verbose)
                failed_segments.append(seg_id)
                continue
        
        if len(segment_metrics) == 0:
            raise ValueError("没有成功评估任何分段")
        
        # 汇总所有分段的平均指标
        valid_segments = len(segment_metrics)
        success_rate = valid_segments / num_segments
        
        log_evaluation_info(f"成功评估: {valid_segments}/{num_segments} 分段 ({success_rate:.1%})", self.verbose)
        
        # 计算最终平均指标
        metrics_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        final_metrics = {}
        
        for name in metrics_names:
            values = [metrics[name] for metrics in segment_metrics]
            final_metrics[name] = compute_statistics(np.array(values))
        
        # 处理信息统计
        scale_ratios = [info['scale_ratio'] for info in segment_info]
        scale_stats = compute_statistics(np.array(scale_ratios))
        
        # 构造结果
        results = {
            'depth_metrics': final_metrics,
            'evaluation_info': {
                'total_frames': num_frames,
                'total_segments': num_segments,
                'valid_segments': valid_segments,
                'success_rate': success_rate,
                'failed_segments': failed_segments,
                'segment_size': segment_size,
                'processing_pipeline': {
                    'min_depth_m': self.min_depth,
                    'max_depth_m': self.max_depth,
                    'resize_to_gt': True,
                    'segmented_median_scale_alignment': True,
                    'depth_range_clipping': True
                }
            },
            'processing_stats': {
                'scale_alignment': scale_stats,
                'avg_frames_per_segment': float(np.mean([info['num_frames'] for info in segment_info]))
            },
            'per_segment_data': {
                'metrics': segment_metrics,
                'processing_info': segment_info
            }
        }
        
        log_evaluation_info("✅ 分段深度评估完成", self.verbose)
        return results

    def evaluate(self, gt_depths: np.ndarray, pred_depths: np.ndarray, mode: str = 'global') -> Dict[str, Any]:
        """
        评估深度序列
        
        Args:
            gt_depths: (N, H, W) GT深度序列，单位m
            pred_depths: (N, H', W') 预测深度序列，单位m
            mode: 评估模式 ('global'=全局中值缩放, 'segmented'=分段中值缩放, 'both'=两种方法)
        
        Returns:
            完整评估结果
        """
        if mode == 'segmented':
            return self.evaluate_segmented(gt_depths, pred_depths)
        elif mode == 'both':
            log_evaluation_info("🎯 开始双重深度评估 (全局 + 分段)", self.verbose)
            
            # 全局评估
            global_results = self.evaluate_global(gt_depths, pred_depths)
            
            # 分段评估  
            segmented_results = self.evaluate_segmented(gt_depths, pred_depths)
            
            # 合并结果
            combined_results = {
                'depth_metrics_global': global_results['depth_metrics'],
                'depth_metrics_segmented': segmented_results['depth_metrics'],
                'evaluation_info': {
                    'mode': 'both',
                    'global_info': global_results['evaluation_info'],
                    'segmented_info': segmented_results['evaluation_info']
                },
                'processing_stats': {
                    'global_stats': global_results['processing_stats'],
                    'segmented_stats': segmented_results['processing_stats']
                }
            }
            
            log_evaluation_info("✅ 双重深度评估完成", self.verbose)
            return combined_results
        else:
            # 默认全局模式
            return self.evaluate_global(gt_depths, pred_depths)
    
    def evaluate_global(self, gt_depths: np.ndarray, pred_depths: np.ndarray) -> Dict[str, Any]:
        """
        全局评估深度序列（原有方法）
        
        Args:
            gt_depths: (N, H, W) GT深度序列，单位m
            pred_depths: (N, H', W') 预测深度序列，单位m
        
        Returns:
            完整评估结果
        """
        log_evaluation_info("🎯 开始全局深度评估", self.verbose)
        
        # 验证输入
        validate_input_shapes(gt_depths, pred_depths, 'depth')
        num_frames = len(gt_depths)
        
        log_evaluation_info(f"总帧数: {num_frames}", self.verbose)
        log_evaluation_info(f"GT尺寸: {gt_depths.shape}", self.verbose)
        log_evaluation_info(f"预测尺寸: {pred_depths.shape}", self.verbose)
        
        # 逐帧评估
        frame_errors = []
        frame_info = []
        skipped_frames = []
        
        for i in tqdm(range(num_frames), desc="评估深度帧", disable=not self.verbose):
            try:
                errors, info = self.evaluate_frame(gt_depths[i], pred_depths[i])
                frame_errors.append(errors)
                frame_info.append(info)
                
            except ValueError as e:
                if self.verbose:
                    log_evaluation_info(f"跳过第{i}帧: {e}", self.verbose)
                skipped_frames.append(i)
                continue
        
        if len(frame_errors) == 0:
            raise ValueError("没有成功评估任何帧")
        
        # 汇总结果
        valid_frames = len(frame_errors)
        success_rate = valid_frames / num_frames
        
        log_evaluation_info(f"成功评估: {valid_frames}/{num_frames} 帧 ({success_rate:.1%})", self.verbose)
        
        # 计算平均指标
        metrics_names = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        
        # 收集每个指标的所有帧数据
        metrics_data = {name: [] for name in metrics_names}
        for errors in frame_errors:
            for name in metrics_names:
                metrics_data[name].append(errors[name])
        
        # 计算统计信息
        final_metrics = {}
        for name in metrics_names:
            values = np.array(metrics_data[name])
            final_metrics[name] = compute_statistics(values)
        
        # 处理信息统计
        scale_ratios = [info['scale_ratio'] for info in frame_info]
        scale_stats = compute_statistics(np.array(scale_ratios))
        
        valid_pixels = [info['valid_pixels'] for info in frame_info]
        final_valid_pixels = [info['final_valid_pixels'] for info in frame_info]
        
        # 构造最终结果
        results = {
            'depth_metrics': final_metrics,
            'evaluation_info': {
                'total_frames': num_frames,
                'valid_frames': valid_frames,
                'success_rate': success_rate,
                'skipped_frames': skipped_frames,
                'processing_pipeline': {
                    'min_depth_m': self.min_depth,
                    'max_depth_m': self.max_depth,
                    'resize_to_gt': True,
                    'median_scale_alignment': True,
                    'depth_range_clipping': True
                }
            },
            'processing_stats': {
                'scale_alignment': scale_stats,
                'avg_valid_pixels': float(np.mean(valid_pixels)),
                'avg_final_valid_pixels': float(np.mean(final_valid_pixels))
            },
            'per_frame_data': {
                'errors': frame_errors,
                'processing_info': frame_info
            }
        }
        
        log_evaluation_info("✅ 全局深度评估完成", self.verbose)
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        打印评估结果摘要
        
        Args:
            results: 评估结果
        """
        print("\n" + "="*70)
        print("🎯 深度评估结果摘要")
        print("="*70)
        
        # 基本信息
        info = results['evaluation_info']
        
        # 检查是否为分段评估结果
        if 'valid_segments' in info:
            # 分段评估结果
            print(f"评估分段数: {info['valid_segments']}/{info['total_segments']} ({info['success_rate']:.1%})")
            print(f"分段大小: {info['segment_size']}帧")
        else:
            # 全局评估结果
            print(f"评估帧数: {info['valid_frames']}/{info['total_frames']} ({info['success_rate']:.1%})")
        
        # 处理统计
        scale_stats = results['processing_stats']['scale_alignment']
        print(f"缩放比例: {scale_stats['mean']:.3f} ± {scale_stats['std']:.3f}")
        
        # 主要指标
        print(f"\n📊 完整指标 (mean ± std):")
        metrics = results['depth_metrics']
        print(f"  abs_rel:  {metrics['abs_rel']['mean']:.3f} ± {metrics['abs_rel']['std']:.3f}")
        print(f"  sq_rel:   {metrics['sq_rel']['mean']*1000:.6f} ± {metrics['sq_rel']['std']*1000:.6f} mm")
        print(f"  rmse:     {metrics['rmse']['mean']*1000:.4f} ± {metrics['rmse']['std']*1000:.4f} mm")
        print(f"  rmse_log: {metrics['rmse_log']['mean']:.3f} ± {metrics['rmse_log']['std']:.3f}")
        print(f"  a1:       {metrics['a1']['mean']:.3f} ± {metrics['a1']['std']:.3f}")
        print(f"  a2:       {metrics['a2']['mean']:.3f} ± {metrics['a2']['std']:.3f}")
        print(f"  a3:       {metrics['a3']['mean']:.3f} ± {metrics['a3']['std']:.3f}")
        
        # 性能评级
        a1_mean = metrics['a1']['mean']
        if a1_mean > 0.8:
            grade = "优秀 (S级)"
        elif a1_mean > 0.6:
            grade = "良好 (A级)"
        else:
            grade = "有待改进 (B级)"
        
        print(f"\n🏆 性能评级: {grade} (a1={a1_mean:.3f})")
        print("="*70)
