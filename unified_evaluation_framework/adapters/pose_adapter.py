"""
位姿数据格式适配器
支持多种位姿数据输入格式，统一转换为标准格式
"""

import os
import numpy as np
import json
from typing import List, Union, Tuple
from tqdm import tqdm

from core.utils import convert_pose_units, log_evaluation_info


class PoseAdapter:
    """
    位姿数据适配器
    
    支持输入格式:
    - npz_dir: 包含.npz文件的目录 (每个文件包含一个位姿)
    - npz_file: 单个.npz文件 (包含所有位姿)
    - txt_file: 文本文件 (每行一个位姿的16个元素)
    - json_file: JSON格式文件
    
    输出标准格式: (N, 4, 4) numpy数组，cam2world格式，位移单位为米(m)
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化位姿适配器
        
        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
        self.supported_formats = ['npz_dir', 'npz_file', 'txt_file', 'json_file']
    
    def load_from_npz_dir(self, input_path: str, input_unit: str = 'm', 
                         pose_key: str = 'pose') -> np.ndarray:
        """
        从.npz文件目录加载位姿数据 (每个文件一个位姿)
        
        Args:
            input_path: 输入目录路径
            input_unit: 输入数据位移单位 ('mm' 或 'm')
            pose_key: 位姿数据的键名
        
        Returns:
            (N, 4, 4) 位姿数组，cam2world格式，位移单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"目录不存在: {input_path}")
        
        # 获取所有.npz文件并排序
        npz_files = [f for f in os.listdir(input_path) if f.endswith('.npz')]
        npz_files.sort()
        
        if len(npz_files) == 0:
            raise ValueError(f"目录中没有找到.npz文件: {input_path}")
        
        log_evaluation_info(f"找到 {len(npz_files)} 个.npz位姿文件", self.verbose)
        
        poses = []
        for filename in tqdm(npz_files, desc="加载.npz位姿", disable=not self.verbose):
            filepath = os.path.join(input_path, filename)
            try:
                data = np.load(filepath)
                
                if pose_key not in data:
                    # 尝试其他可能的键名
                    possible_keys = ['pose', 'poses', 'transformation', 'T']
                    found_key = None
                    for key in possible_keys:
                        if key in data:
                            found_key = key
                            break
                    
                    if found_key is None:
                        log_evaluation_info(f"{filename} 中没有找到位姿数据键", self.verbose)
                        continue
                    else:
                        pose_key = found_key
                
                pose = data[pose_key].astype(np.float64)
                
                # 验证位姿矩阵形状
                if pose.shape != (4, 4):
                    log_evaluation_info(f"{filename} 位姿形状错误: {pose.shape}", self.verbose)
                    continue
                
                poses.append(pose)
                
            except Exception as e:
                log_evaluation_info(f"加载 {filename} 失败: {e}", self.verbose)
                continue
        
        if len(poses) == 0:
            raise ValueError("没有成功加载任何位姿文件")
        
        poses = np.array(poses)
        
        # 单位转换
        poses = convert_pose_units(poses, input_unit, 'm')
        
        log_evaluation_info(f"成功加载位姿: {poses.shape}, 格式: cam2world, 位移单位: m", self.verbose)
        return poses
    
    def load_from_npz_file(self, input_path: str, input_unit: str = 'm', 
                          data_key: str = None) -> np.ndarray:
        """
        从单个.npz文件加载位姿数据 (包含所有位姿)
        """
        # --- 根本性修复 ---
        # 如果上游错误地将目录路径传给了这个函数，则自动切换到目录加载模式
        if os.path.isdir(input_path):
            log_evaluation_info(f"警告: 'load_from_npz_file' 接收到目录路径. 自动切换到 'npz_dir' 加载模式.", self.verbose)
            return self.load_from_npz_dir(input_path, input_unit)
        # --- 修复结束 ---

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        try:
            data = np.load(input_path)
            
            # 自动检测数据键
            if data_key is None:
                possible_keys = ['poses', 'data', 'transformations', 'pose']
                for key in possible_keys:
                    if key in data:
                        data_key = key
                        break
                
                if data_key is None:
                    # 使用第一个数组
                    data_key = list(data.keys())[0]
                    log_evaluation_info(f"使用第一个数组键: {data_key}", self.verbose)
            
            poses = data[data_key].astype(np.float64)
            
            # 验证形状
            if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
                raise ValueError(f"位姿数据形状错误: {poses.shape}, 期望: (N, 4, 4)")
            
        except Exception as e:
            raise ValueError(f"加载NPZ文件失败: {e}")
        
        # 单位转换
        poses = convert_pose_units(poses, input_unit, 'm')
        
        log_evaluation_info(f"成功加载位姿: {poses.shape}, 键: {data_key}, 位移单位: m", self.verbose)
        return poses
    
    def load_from_txt_file(self, input_path: str, input_unit: str = 'm', 
                          format_type: str = 'matrix') -> np.ndarray:
        """
        从文本文件加载位姿数据
        
        Args:
            input_path: 输入文件路径
            input_unit: 输入数据位移单位 ('mm' 或 'm')
            format_type: 文件格式类型
                - 'matrix': 每行16个元素 (4x4矩阵按行展开)
                - 'kitti': KITTI格式 (3x4矩阵，每行12个元素)
        
        Returns:
            (N, 4, 4) 位姿数组，cam2world格式，位移单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        try:
            # 读取文本数据
            data = np.loadtxt(input_path)
            
            if format_type == 'matrix':
                # 每行16个元素 (4x4矩阵)
                if data.shape[1] != 16:
                    raise ValueError(f"期望每行16个元素，得到: {data.shape[1]}")
                
                poses = data.reshape(-1, 4, 4)
                
            elif format_type == 'kitti':
                # KITTI格式：每行12个元素 (3x4矩阵)
                if data.shape[1] != 12:
                    raise ValueError(f"KITTI格式期望每行12个元素，得到: {data.shape[1]}")
                
                num_poses = data.shape[0]
                poses = np.zeros((num_poses, 4, 4))
                
                for i in range(num_poses):
                    pose_3x4 = data[i].reshape(3, 4)
                    poses[i, :3, :] = pose_3x4
                    poses[i, 3, 3] = 1.0  # 齐次坐标
            
            else:
                raise ValueError(f"不支持的文本格式: {format_type}")
            
        except Exception as e:
            raise ValueError(f"加载文本文件失败: {e}")
        
        # 单位转换
        poses = convert_pose_units(poses, input_unit, 'm')
        
        log_evaluation_info(f"成功加载位姿: {poses.shape}, 格式: {format_type}, 位移单位: m", self.verbose)
        return poses
    
    def load_from_json_file(self, input_path: str, input_unit: str = 'm') -> np.ndarray:
        """
        从JSON文件加载位姿数据
        
        Args:
            input_path: 输入文件路径
            input_unit: 输入数据位移单位 ('mm' 或 'm')
        
        Returns:
            (N, 4, 4) 位姿数组，cam2world格式，位移单位m
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # 尝试不同的数据结构
            poses_list = None
            
            if isinstance(data, list):
                # 直接是位姿列表
                poses_list = data
            elif isinstance(data, dict):
                # 字典格式，查找位姿数据
                possible_keys = ['poses', 'transformations', 'cameras', 'frames']
                for key in possible_keys:
                    if key in data:
                        poses_list = data[key]
                        break
                
                if poses_list is None:
                    raise ValueError("JSON文件中没有找到位姿数据")
            
            # 转换为numpy数组
            poses = []
            for pose_data in poses_list:
                if isinstance(pose_data, list):
                    # 扁平化列表
                    if len(pose_data) == 16:
                        pose = np.array(pose_data).reshape(4, 4)
                    elif len(pose_data) == 12:
                        # 3x4格式
                        pose = np.eye(4)
                        pose[:3, :] = np.array(pose_data).reshape(3, 4)
                    else:
                        raise ValueError(f"不支持的位姿数据长度: {len(pose_data)}")
                elif isinstance(pose_data, dict):
                    # 字典格式 (可能包含'matrix', 'transformation'等键)
                    if 'matrix' in pose_data:
                        pose = np.array(pose_data['matrix'])
                    elif 'transformation' in pose_data:
                        pose = np.array(pose_data['transformation'])
                    else:
                        raise ValueError("字典中没有找到位姿矩阵")
                else:
                    raise ValueError(f"不支持的位姿数据类型: {type(pose_data)}")
                
                poses.append(pose)
            
            poses = np.array(poses).astype(np.float64)
            
        except Exception as e:
            raise ValueError(f"加载JSON文件失败: {e}")
        
        # 单位转换
        poses = convert_pose_units(poses, input_unit, 'm')
        
        log_evaluation_info(f"成功加载位姿: {poses.shape}, 格式: JSON, 位移单位: m", self.verbose)
        return poses
    
    def load(self, input_path: str, input_format: str, input_unit: str = 'm', 
            **kwargs) -> np.ndarray:
        """
        通用位姿数据加载接口
        
        Args:
            input_path: 输入路径
            input_format: 输入格式 ('npz_dir', 'npz_file', 'txt_file', 'json_file')
            input_unit: 输入数据位移单位 ('mm' 或 'm')
            **kwargs: 格式特定的参数
        
        Returns:
            (N, 4, 4) 标准化位姿数组，cam2world格式，位移单位m
        """
        if input_format not in self.supported_formats:
            raise ValueError(f"不支持的格式: {input_format}. 支持的格式: {self.supported_formats}")
        
        log_evaluation_info(f"加载位姿数据: {input_path} ({input_format})", self.verbose)
        
        if input_format == 'npz_dir':
            return self.load_from_npz_dir(input_path, input_unit, **kwargs)
        elif input_format == 'npz_file':
            return self.load_from_npz_file(input_path, input_unit, **kwargs)
        elif input_format == 'txt_file':
            return self.load_from_txt_file(input_path, input_unit, **kwargs)
        elif input_format == 'json_file':
            return self.load_from_json_file(input_path, input_unit, **kwargs)
    
    def validate_poses(self, poses: np.ndarray) -> dict:
        """
        验证位姿数据质量
        
        Args:
            poses: (N, 4, 4) 位姿数组
        
        Returns:
            验证结果
        """
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
            return {'valid': False, 'error': f'Expected (N, 4, 4), got {poses.shape}'}
        
        num_poses = len(poses)
        warnings = []
        
        # 检查齐次坐标
        bottom_rows = poses[:, 3, :]
        expected_bottom = np.array([0, 0, 0, 1])
        
        valid_bottom = np.allclose(bottom_rows, expected_bottom, atol=1e-6)
        if not valid_bottom:
            warnings.append("Bottom row is not [0, 0, 0, 1]")
        
        # 检查旋转矩阵的有效性
        rotation_matrices = poses[:, :3, :3]
        det_R = np.linalg.det(rotation_matrices)
        valid_det = np.allclose(det_R, 1.0, atol=1e-3)
        
        if not valid_det:
            invalid_count = np.sum(~np.isclose(det_R, 1.0, atol=1e-3))
            warnings.append(f"{invalid_count}/{num_poses} poses have invalid rotation matrices (det != 1)")
        
        # 检查位移范围
        translations = poses[:, :3, 3]
        translation_norms = np.linalg.norm(translations, axis=1)
        
        translation_stats = {
            'min': float(np.min(translation_norms)),
            'max': float(np.max(translation_norms)),
            'mean': float(np.mean(translation_norms)),
            'std': float(np.std(translation_norms))
        }
        
        # 检查异常值
        if translation_stats['max'] > 1000:
            warnings.append(f"Very large translation detected: {translation_stats['max']:.1f}m")
        
        if translation_stats['min'] == translation_stats['max']:
            warnings.append("All poses have identical translations (static camera)")
        
        result = {
            'valid': len(warnings) == 0 or valid_bottom,  # 至少齐次坐标要正确
            'num_poses': num_poses,
            'translation_stats': translation_stats,
            'rotation_matrix_valid': valid_det,
            'homogeneous_coords_valid': valid_bottom,
            'warnings': warnings
        }
        
        return result
    
    def convert_world2cam_to_cam2world(self, poses: np.ndarray) -> np.ndarray:
        """
        将world2cam格式转换为cam2world格式
        
        Args:
            poses: (N, 4, 4) world2cam位姿数组
        
        Returns:
            (N, 4, 4) cam2world位姿数组
        """
        cam2world_poses = np.array([np.linalg.inv(pose) for pose in poses])
        
        log_evaluation_info(f"转换位姿格式: world2cam -> cam2world", self.verbose)
        return cam2world_poses
    
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
            
            elif input_path.endswith(('.txt', '.dat')):
                info['suggested_format'] = 'txt_file'
                info['type'] = 'file'
                try:
                    data = np.loadtxt(input_path)
                    info['shape'] = data.shape
                    if data.shape[1] == 16:
                        info['detected_format'] = 'matrix'
                    elif data.shape[1] == 12:
                        info['detected_format'] = 'kitti'
                    else:
                        info['warning'] = f'Unexpected column count: {data.shape[1]}'
                except:
                    info['error'] = 'Cannot load text file'
            
            elif input_path.endswith('.json'):
                info['suggested_format'] = 'json_file'
                info['type'] = 'file'
                try:
                    with open(input_path, 'r') as f:
                        data = json.load(f)
                    info['json_structure'] = type(data).__name__
                    if isinstance(data, dict):
                        info['keys'] = list(data.keys())
                except:
                    info['error'] = 'Cannot load JSON file'
            
            else:
                info['type'] = 'file'
                info['error'] = 'Unsupported file format'
        
        elif os.path.isdir(input_path):
            # 目录
            files = os.listdir(input_path)
            npz_files = [f for f in files if f.endswith('.npz')]
            
            info['type'] = 'directory'
            info['file_counts'] = {
                'npz': len(npz_files),
                'total': len(files)
            }
            
            if len(npz_files) > 0:
                info['suggested_format'] = 'npz_dir'
                # 检查一个文件的内容
                try:
                    sample_file = os.path.join(input_path, npz_files[0])
                    data = np.load(sample_file)
                    info['sample_keys'] = list(data.keys())
                    info['sample_shapes'] = {k: data[k].shape for k in data.keys()}
                except:
                    info['warning'] = 'Cannot inspect sample NPZ file'
            else:
                info['error'] = 'No supported files found'
        
        return info
