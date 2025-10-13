"""
统一评估框架通用工具函数
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
import json
from datetime import datetime


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def convert_depth_units(depth: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """
    深度单位转换
    
    Args:
        depth: 输入深度数组
        from_unit: 原单位 ('mm', 'm')
        to_unit: 目标单位 ('mm', 'm')
    
    Returns:
        转换后的深度数组
    """
    if from_unit == to_unit:
        return depth.copy()
    
    if from_unit == 'mm' and to_unit == 'm':
        return depth / 1000.0
    elif from_unit == 'm' and to_unit == 'mm':
        return depth * 1000.0
    else:
        raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")


def convert_pose_units(poses: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """
    位姿位移单位转换
    
    Args:
        poses: (N, 4, 4) 位姿矩阵
        from_unit: 原单位 ('mm', 'm')
        to_unit: 目标单位 ('mm', 'm')
    
    Returns:
        转换后的位姿矩阵
    """
    if from_unit == to_unit:
        return poses.copy()
    
    poses_converted = poses.copy()
    
    if from_unit == 'mm' and to_unit == 'm':
        poses_converted[:, :3, 3] /= 1000.0
    elif from_unit == 'm' and to_unit == 'mm':
        poses_converted[:, :3, 3] *= 1000.0
    else:
        raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")
    
    return poses_converted


def resize_depth(depth: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整深度图尺寸
    
    Args:
        depth: (H, W) 深度图
        target_size: (H, W) 目标尺寸
    
    Returns:
        调整后的深度图
    """
    if depth.shape == target_size:
        return depth.copy()
    
    # OpenCV的resize需要(width, height)顺序
    target_width, target_height = target_size[1], target_size[0]
    resized = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def create_depth_mask(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                     min_depth: float = 1e-3, max_depth: float = 150.0) -> np.ndarray:
    """
    创建深度有效区域掩码
    
    Args:
        gt_depth: GT深度图 (单位: m)
        pred_depth: 预测深度图 (单位: m)
        min_depth: 最小有效深度 (m)
        max_depth: 最大有效深度 (m)
    
    Returns:
        有效区域掩码
    """
    # GT深度有效性检查
    gt_valid = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    gt_valid = np.logical_and(gt_valid, np.isfinite(gt_depth))
    
    # 预测深度有效性检查
    pred_valid = np.logical_and(pred_depth > 0, np.isfinite(pred_depth))
    
    # 合并掩码
    mask = np.logical_and(gt_valid, pred_valid)
    
    return mask


def median_scale_alignment(gt_depth: np.ndarray, pred_depth: np.ndarray, 
                          mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    中值缩放对齐
    
    Args:
        gt_depth: GT深度图
        pred_depth: 预测深度图
        mask: 有效区域掩码
    
    Returns:
        对齐后的预测深度, 缩放比例
    """
    if np.sum(mask) < 100:
        raise ValueError("有效像素太少，无法进行缩放对齐")
    
    gt_masked = gt_depth[mask]
    pred_masked = pred_depth[mask]
    
    gt_median = np.median(gt_masked)
    pred_median = np.median(pred_masked)
    
    if pred_median <= 0:
        raise ValueError("预测深度中值非正，无法进行缩放")
    
    ratio = gt_median / pred_median
    pred_scaled = pred_depth * ratio
    
    return pred_scaled, ratio


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """
    保存评估结果到JSON文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    ensure_dir(os.path.dirname(output_path))
    
    # 添加时间戳
    results['evaluation_time'] = datetime.now().isoformat()
    
    # 自定义JSON编码器，处理numpy数组
    def json_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')
    
    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=json_serializer)


def compute_statistics(errors: np.ndarray) -> Dict[str, float]:
    """
    计算误差统计信息
    
    Args:
        errors: 误差数组
    
    Returns:
        统计信息字典
    """
    return {
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'median': float(np.median(errors)),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'min': float(np.min(errors)),
        'max': float(np.max(errors))
    }


def validate_input_shapes(gt_data: np.ndarray, pred_data: np.ndarray, data_type: str) -> None:
    """
    验证输入数据形状
    
    Args:
        gt_data: GT数据
        pred_data: 预测数据  
        data_type: 数据类型 ('depth' 或 'pose')
    """
    if len(gt_data) != len(pred_data):
        raise ValueError(f"{data_type}数据帧数不匹配: GT={len(gt_data)}, Pred={len(pred_data)}")
    
    if data_type == 'depth':
        if gt_data.ndim != 3 or pred_data.ndim != 3:
            raise ValueError(f"深度数据应为3维 (N, H, W), 得到: GT={gt_data.shape}, Pred={pred_data.shape}")
    elif data_type == 'pose':
        if gt_data.shape != pred_data.shape or gt_data.shape[-2:] != (4, 4):
            raise ValueError(f"位姿数据应为 (N, 4, 4), 得到: GT={gt_data.shape}, Pred={pred_data.shape}")


def log_evaluation_info(info: str, verbose: bool = True) -> None:
    """
    记录评估信息
    
    Args:
        info: 信息内容
        verbose: 是否打印到控制台
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {info}"
    
    if verbose:
        print(log_line)
