"""
深度数据格式适配器
支持多种深度数据输入格式，统一转换为标准格式
"""

import os
import numpy as np
import cv2
from typing import List, Union, Tuple
from tqdm import tqdm

from core.utils import convert_depth_units, log_evaluation_info


class DepthAdapter:
    """
    深度数据适配器
    
    支持输入格式:
    - npy_dir: 包含.npy文件的目录
    - npz_file: 单个.npz文件
    - tiff_dir: 包含.tiff文件的目录
    
    输出标准格式: (N, H, W) numpy数组，单位为米(m)
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化深度适配器
        
        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.supported_formats = ['npy_dir', 'npz_file', 'tiff_dir']
    
    def load_from_npy_dir(self, input_path: str, input_unit: str = 'm') -> np.ndarray:
        """
        从.npy文件目录加载深度数据
        
        Args:
            input_path: 输入目录路径
            input_unit: 输入数据单位 ('mm' 或 'm')
        
        Returns:
            (N, H, W) 深度数组，单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"目录不存在: {input_path}")
        
        # 获取所有.npy文件并排序
        npy_files = [f for f in os.listdir(input_path) if f.endswith('.npy')]
        npy_files.sort()
        
        if len(npy_files) == 0:
            raise ValueError(f"目录中没有找到.npy文件: {input_path}")
        
        log_evaluation_info(f"找到 {len(npy_files)} 个.npy文件", self.verbose)
        
        depths = []
        for filename in tqdm(npy_files, desc="加载.npy深度", disable=not self.verbose):
            filepath = os.path.join(input_path, filename)
            try:
                depth = np.load(filepath).astype(np.float32)
                depths.append(depth)
            except Exception as e:
                log_evaluation_info(f"加载 {filename} 失败: {e}", self.verbose)
                continue
        
        if len(depths) == 0:
            raise ValueError("没有成功加载任何深度文件")
        
        depths = np.array(depths)
        
        # 单位转换
        depths = convert_depth_units(depths, input_unit, 'm')
        
        log_evaluation_info(f"成功加载深度: {depths.shape}, 单位: m", self.verbose)
        return depths
    
    def load_from_npz_file(self, input_path: str, input_unit: str = 'm', 
                          data_key: str = None) -> np.ndarray:
        """
        从.npz文件加载深度数据
        
        Args:
            input_path: 输入文件路径
            input_unit: 输入数据单位 ('mm' 或 'm')
            data_key: 数据键名，如果None则自动检测
        
        Returns:
            (N, H, W) 深度数组，单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        try:
            data = np.load(input_path)
            
            # 自动检测数据键
            if data_key is None:
                possible_keys = ['depths', 'data', 'depth']
                for key in possible_keys:
                    if key in data:
                        data_key = key
                        break
                
                if data_key is None:
                    # 使用第一个数组
                    data_key = list(data.keys())[0]
                    log_evaluation_info(f"使用第一个数组键: {data_key}", self.verbose)
            
            depths = data[data_key].astype(np.float32)
            
        except Exception as e:
            raise ValueError(f"加载NPZ文件失败: {e}")
        
        # 单位转换
        depths = convert_depth_units(depths, input_unit, 'm')
        
        log_evaluation_info(f"成功加载深度: {depths.shape}, 键: {data_key}, 单位: m", self.verbose)
        return depths
    
    def load_from_tiff_dir(self, input_path: str, input_unit: str = 'mm', 
                          crop_rows: int = None) -> np.ndarray:
        """
        从.tiff文件目录加载深度数据
        
        Args:
            input_path: 输入目录路径
            input_unit: 输入数据单位 ('mm' 或 'm')
            crop_rows: 如果指定，则裁剪前N行
        
        Returns:
            (N, H, W) 深度数组，单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"目录不存在: {input_path}")
        
        # 获取所有TIFF文件并排序
        tiff_files = [f for f in os.listdir(input_path) if f.endswith(('.tiff', '.tif'))]
        tiff_files.sort()
        
        if len(tiff_files) == 0:
            raise ValueError(f"目录中没有找到TIFF文件: {input_path}")
        
        log_evaluation_info(f"找到 {len(tiff_files)} 个TIFF文件", self.verbose)
        
        depths = []
        for filename in tqdm(tiff_files, desc="加载TIFF深度", disable=not self.verbose):
            filepath = os.path.join(input_path, filename)
            try:
                # 读取TIFF文件
                depth = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                
                if depth is None:
                    log_evaluation_info(f"无法读取 {filename}", self.verbose)
                    continue
                
                # 如果是3通道图像，提取第一通道
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                
                # 裁剪处理
                if crop_rows is not None:
                    depth = depth[:crop_rows, :]
                
                depth = depth.astype(np.float32)
                depths.append(depth)
                
            except Exception as e:
                log_evaluation_info(f"加载 {filename} 失败: {e}", self.verbose)
                continue
        
        if len(depths) == 0:
            raise ValueError("没有成功加载任何深度文件")
        
        depths = np.array(depths)
        
        # 单位转换
        depths = convert_depth_units(depths, input_unit, 'm')
        
        log_evaluation_info(f"成功加载深度: {depths.shape}, 单位: m", self.verbose)
        return depths
    
    def load(self, input_path: str, input_format: str, input_unit: str = 'm', 
            **kwargs) -> np.ndarray:
        """
        通用深度数据加载接口
        
        Args:
            input_path: 输入路径
            input_format: 输入格式 ('npy_dir', 'npz_file', 'tiff_dir')
            input_unit: 输入数据单位 ('mm' 或 'm')
            **kwargs: 格式特定的参数
        
        Returns:
            (N, H, W) 标准化深度数组，单位m
        """
        if input_format not in self.supported_formats:
            raise ValueError(f"不支持的格式: {input_format}. 支持的格式: {self.supported_formats}")
        
        log_evaluation_info(f"加载深度数据: {input_path} ({input_format})", self.verbose)
        
        if input_format == 'npy_dir':
            return self.load_from_npy_dir(input_path, input_unit)
        elif input_format == 'npz_file':
            return self.load_from_npz_file(input_path, input_unit, **kwargs)
        elif input_format == 'tiff_dir':
            return self.load_from_tiff_dir(input_path, input_unit, **kwargs)
    
    def get_format_info(self, input_path: str) -> dict:
        """
        自动检测输入数据格式信息
        
        Args:
            input_path: 输入路径
        
        Returns:
            格式信息字典
        """
        if not os.path.exists(input_path):
            return {'exists': False, 'error': 'Path does not exist'}
        
        info = {'exists': True, 'path': input_path}
        
        if os.path.isfile(input_path):
            # 文件
            if input_path.endswith('.npz'):
                info['suggested_format'] = 'npz_file'
                info['type'] = 'file'
                try:
                    data = np.load(input_path)
                    info['keys'] = list(data.keys())
                    info['shapes'] = {k: data[k].shape for k in data.keys()}
                except:
                    info['error'] = 'Cannot load NPZ file'
            else:
                info['type'] = 'file'
                info['error'] = 'Unsupported file format'
        
        elif os.path.isdir(input_path):
            # 目录
            files = os.listdir(input_path)
            npy_files = [f for f in files if f.endswith('.npy')]
            tiff_files = [f for f in files if f.endswith(('.tiff', '.tif'))]
            
            info['type'] = 'directory'
            info['file_counts'] = {
                'npy': len(npy_files),
                'tiff': len(tiff_files),
                'total': len(files)
            }
            
            if len(npy_files) > 0:
                info['suggested_format'] = 'npy_dir'
            elif len(tiff_files) > 0:
                info['suggested_format'] = 'tiff_dir'
            else:
                info['error'] = 'No supported files found'
        
        return info
    
    def validate_data(self, depths: np.ndarray) -> dict:
        """
        验证深度数据质量
        
        Args:
            depths: (N, H, W) 深度数组
        
        Returns:
            验证结果
        """
        if depths.ndim != 3:
            return {'valid': False, 'error': f'Expected 3D array, got {depths.ndim}D'}
        
        # 基本统计
        valid_pixels = np.isfinite(depths) & (depths > 0)
        valid_ratio = np.mean(valid_pixels)
        
        depth_range = (float(np.min(depths[valid_pixels])), float(np.max(depths[valid_pixels])))
        
        result = {
            'valid': True,
            'shape': depths.shape,
            'valid_pixel_ratio': float(valid_ratio),
            'depth_range_m': depth_range,
            'num_frames': len(depths),
            'frame_resolution': depths.shape[1:],
            'dtype': str(depths.dtype)
        }
        
        # 检查潜在问题
        warnings = []
        if valid_ratio < 0.1:
            warnings.append(f"Valid pixel ratio very low: {valid_ratio:.1%}")
        
        if depth_range[1] > 1000:
            warnings.append(f"Suspicious max depth: {depth_range[1]:.1f}m (might be in mm?)")
        
        if depth_range[1] < 0.01:
            warnings.append(f"Suspicious small depth range: {depth_range}")
        
        result['warnings'] = warnings
        
        return result
