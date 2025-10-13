#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScanNet数据集预处理脚本 - 详细注释版本
==================================
本文件是对 preprocess_scannet.py 的详细注释版本
每一行代码都有对应的中文解释，帮助理解数据预处理的完整流程

ScanNet数据集预处理主要功能：
1. 读取RGB图像、深度图、相机位姿和内参
2. 将数据重组为标准的训练格式
3. 输出到指定目录，供深度学习模型训练使用

输入数据结构：
scannet_dir/
├── scans_train/          # 训练集
│   └── scene0000_00/     # 场景目录
│       ├── color/        # RGB图像目录
│       ├── depth/        # 深度图目录 
│       ├── pose/         # 相机位姿目录
│       └── intrinsic/    # 相机内参目录
└── scans_test/           # 测试集
    └── ...

输出数据结构：
output_dir/
├── scans_train/
│   └── scene0000_00/
│       ├── color/        # 处理后的RGB图像
│       ├── depth/        # 处理后的深度图
│       └── cam/          # 相机参数(.npz格式)
└── scans_test/
    └── ...
"""

# ============================================================================
# 导入必要的库 (Import Required Libraries)
# ============================================================================

import argparse          # 用于解析命令行参数，让脚本可以接收用户输入的路径等参数
import random            # 用于生成随机数，虽然在这个脚本中没有直接使用
import gzip              # 用于处理压缩文件，虽然在这个脚本中没有直接使用
import json              # 用于处理JSON格式文件，虽然在这个脚本中没有直接使用
import os                # 提供操作系统接口，用于文件和目录操作
import os.path as osp    # 给os.path起别名，简化路径操作的写法

import torch             # PyTorch深度学习框架，虽然在这个脚本中没有直接使用
import PIL.Image         # Python图像库，用于图像处理操作
from PIL import Image    # 从PIL导入Image类，用于读取和保存图像
import numpy as np       # 数值计算库，用于处理数组和矩阵运算
import cv2               # OpenCV计算机视觉库，用于图像和深度图的读写
import multiprocessing   # 多进程处理库，用于并行处理多个场景，提高处理速度
from tqdm import tqdm    # 进度条库，用于显示处理进度
import matplotlib.pyplot as plt  # 绘图库，虽然在这个脚本中没有直接使用
import shutil            # 文件操作库，虽然在这个脚本中没有直接使用
import path_to_root      # 自定义模块，用于设置项目根路径，noqa表示忽略lint检查
import datasets_preprocess.utils.cropping as cropping  # 自定义裁剪工具模块，虽然在这个脚本中没有直接使用


# ============================================================================
# 命令行参数解析函数 (Command Line Argument Parser)
# ============================================================================

def get_parser():
    """
    创建并配置命令行参数解析器
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    # 创建ArgumentParser对象，用于定义程序接受的命令行参数
    parser = argparse.ArgumentParser()
    
    # 添加ScanNet数据集路径参数
    # --scannet_dir: 指定原始ScanNet数据集的根目录路径
    # default: 设置默认值为"data/data_scannet"
    parser.add_argument("--scannet_dir", default="data/data_scannet")
    
    # 添加输出目录路径参数  
    # --output_dir: 指定处理后数据的输出目录路径
    # default: 设置默认值为"data/dust3r_data/processed_scannet"
    parser.add_argument("--output_dir", default="data/dust3r_data/processed_scannet")
    
    # 返回配置好的解析器对象
    return parser


# ============================================================================
# 场景处理函数 (Scene Processing Function)
# ============================================================================

def process_scene(args):
    """
    处理单个场景的所有帧数据
    
    这个函数是整个预处理的核心，负责：
    1. 读取RGB图像、深度图、位姿和内参
    2. 验证数据有效性
    3. 重新格式化并保存到输出目录
    
    Args:
        args (tuple): 包含(rootdir, outdir, split, scene)的元组
            - rootdir: 原始数据根目录
            - outdir: 输出数据根目录  
            - split: 数据集分割名称(如"scans_train"或"scans_test")
            - scene: 场景名称(如"scene0000_00")
    """
    # 解包参数元组，获取各个路径和标识符
    rootdir, outdir, split, scene = args
    
    # ========================================================================
    # 构建输入路径 (Build Input Paths)
    # ========================================================================
    
    # 构建当前场景的根目录路径
    # 例如: /path/to/scannet/scans_train/scene0000_00
    frame_dir = osp.join(rootdir, split, scene)
    
    # 构建RGB图像目录路径
    # 例如: /path/to/scannet/scans_train/scene0000_00/color
    rgb_dir = osp.join(frame_dir, "color")
    
    # 构建深度图目录路径  
    # 例如: /path/to/scannet/scans_train/scene0000_00/depth
    depth_dir = osp.join(frame_dir, "depth")
    
    # 构建相机位姿目录路径
    # 例如: /path/to/scannet/scans_train/scene0000_00/pose  
    pose_dir = osp.join(frame_dir, "pose")
    
    # ========================================================================
    # 读取相机内参 (Load Camera Intrinsics)
    # ========================================================================
    
    # 读取深度相机内参矩阵
    # loadtxt: 从文本文件加载数据
    # [:3, :3]: 取前3行3列，得到3x3的内参矩阵K
    # astype(np.float32): 转换为32位浮点数类型
    depth_intrinsic = np.loadtxt(
        osp.join(frame_dir, "intrinsic", "intrinsic_depth.txt")
    )[:3, :3].astype(np.float32)
    
    # 读取RGB相机内参矩阵（处理方式同深度相机）
    color_intrinsic = np.loadtxt(
        osp.join(frame_dir, "intrinsic", "intrinsic_color.txt")  
    )[:3, :3].astype(np.float32)
    
    # 验证内参矩阵的有效性
    # isfinite(): 检查数组中所有元素是否为有限数值（非无穷大、非NaN）
    # 如果内参矩阵包含无效值，则跳过这个场景
    if not np.isfinite(depth_intrinsic).all() or not np.isfinite(color_intrinsic).all():
        return  # 提前退出函数，不处理这个场景
    
    # ========================================================================
    # 创建输出目录 (Create Output Directories)
    # ========================================================================
    
    # 创建当前场景的输出根目录
    # exist_ok=True: 如果目录已存在不会报错
    os.makedirs(osp.join(outdir, split, scene), exist_ok=True)
    
    # 统计当前场景的帧数
    # 通过RGB目录中的文件数量来确定总帧数
    frame_num = len(os.listdir(rgb_dir))
    
    # 验证数据完整性：确保RGB、深度、位姿目录中的文件数量一致
    # 如果数量不一致，程序会抛出AssertionError并终止
    assert frame_num == len(os.listdir(depth_dir)) == len(os.listdir(pose_dir))
    
    # 构建输出子目录路径
    out_rgb_dir = osp.join(outdir, split, scene, "color")      # RGB输出目录
    out_depth_dir = osp.join(outdir, split, scene, "depth")    # 深度输出目录  
    out_cam_dir = osp.join(outdir, split, scene, "cam")        # 相机参数输出目录

    # 创建所有输出子目录
    os.makedirs(out_rgb_dir, exist_ok=True)     # 创建RGB输出目录
    os.makedirs(out_depth_dir, exist_ok=True)   # 创建深度输出目录
    os.makedirs(out_cam_dir, exist_ok=True)     # 创建相机参数输出目录
    
    # ========================================================================
    # 逐帧处理 (Frame-by-Frame Processing)
    # ========================================================================
    
    # 使用tqdm显示进度条，遍历所有帧
    for i in tqdm(range(frame_num)):
        # ====================================================================
        # 构建当前帧的输入文件路径 (Build Input File Paths)
        # ====================================================================
        
        # 构建RGB图像文件路径，ScanNet中RGB图像以.jpg格式存储
        # 例如: /path/to/scannet/scans_train/scene0000_00/color/0.jpg
        rgb_path = osp.join(rgb_dir, f"{i}.jpg")
        
        # 构建深度图文件路径，ScanNet中深度图以.png格式存储
        # 例如: /path/to/scannet/scans_train/scene0000_00/depth/0.png  
        depth_path = osp.join(depth_dir, f"{i}.png")
        
        # 构建相机位姿文件路径，位姿以.txt格式存储
        # 例如: /path/to/scannet/scans_train/scene0000_00/pose/0.txt
        pose_path = osp.join(pose_dir, f"{i}.txt")

        # ====================================================================
        # 读取当前帧数据 (Load Current Frame Data)
        # ====================================================================
        
        # 使用PIL读取RGB图像
        # Image.open(): PIL的图像读取函数，返回PIL Image对象
        rgb = Image.open(rgb_path)
        
        # 使用OpenCV读取深度图
        # cv2.IMREAD_UNCHANGED: 保持原始数据类型和通道数不变
        # ScanNet的深度图通常是16位无符号整数格式
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        # 调整RGB图像尺寸以匹配深度图
        # depth.shape[::-1]: 将深度图的(H,W)转换为(W,H)，因为PIL需要(width, height)格式
        # resample=Image.Resampling.LANCZOS: 使用LANCZOS重采样算法，质量较高
        rgb = rgb.resize(depth.shape[::-1], resample=Image.Resampling.LANCZOS)
        
        # 读取相机位姿矩阵
        # loadtxt(): 从文本文件读取数值数据
        # reshape(4, 4): 将一维数组重塑为4x4矩阵（齐次变换矩阵）
        # astype(np.float32): 转换为32位浮点数
        pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        
        # 验证位姿矩阵的有效性
        # 如果位姿矩阵包含无穷大或NaN值，跳过当前帧
        if not np.isfinite(pose).all():
            continue  # 跳过当前帧，处理下一帧

        # ====================================================================
        # 构建输出文件路径 (Build Output File Paths)  
        # ====================================================================
        
        # 构建RGB输出路径，使用5位零填充的帧编号
        # f"{i:05d}": 将帧编号格式化为5位数字，不足位数用0填充
        # 例如: 第0帧变为"00000"，第100帧变为"00100"
        out_rgb_path = osp.join(out_rgb_dir, f"{i:05d}.jpg")
        
        # 构建深度图输出路径
        out_depth_path = osp.join(out_depth_dir, f"{i:05d}.png")
        
        # 构建相机参数输出路径，使用.npz格式（NumPy压缩格式）
        out_cam_path = osp.join(out_cam_dir, f"{i:05d}.npz")
        
        # ====================================================================
        # 保存处理后的数据 (Save Processed Data)
        # ====================================================================
        
        # 保存相机参数到.npz文件
        # np.savez(): 将多个数组保存到一个压缩的.npz文件中
        # intrinsics=depth_intrinsic: 保存深度相机内参矩阵
        # pose=pose: 保存相机位姿矩阵
        np.savez(out_cam_path, intrinsics=depth_intrinsic, pose=pose)
        
        # 保存RGB图像
        # PIL Image对象的save方法，保存为JPEG格式
        rgb.save(out_rgb_path)
        
        # 保存深度图
        # cv2.imwrite(): OpenCV的图像写入函数
        # 保持原始深度值和数据格式
        cv2.imwrite(out_depth_path, depth)


# ============================================================================
# 主处理函数 (Main Processing Function)
# ============================================================================

def main(rootdir, outdir):
    """
    主函数：处理整个ScanNet数据集
    
    这个函数负责：
    1. 设置输出目录
    2. 遍历所有数据集分割（训练集、测试集）
    3. 使用多进程并行处理所有场景
    
    Args:
        rootdir (str): 原始ScanNet数据集根目录路径
        outdir (str): 处理后数据输出根目录路径
    """
    # 创建输出根目录
    # exist_ok=True: 如果目录已存在不会报错
    os.makedirs(outdir, exist_ok=True)
    
    # 定义要处理的数据集分割
    # ScanNet数据集包含训练集和测试集两个部分
    splits = ["scans_test", "scans_train"]
    
    # 创建多进程池
    # processes=multiprocessing.cpu_count(): 使用CPU核心数作为进程数
    # 这样可以充分利用多核CPU的并行处理能力
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # 遍历每个数据集分割
    for split in splits:
        # ====================================================================
        # 获取当前分割中的所有场景 (Get All Scenes in Current Split)
        # ====================================================================
        
        # 列出当前分割目录下的所有子目录（场景）
        # os.listdir(): 列出目录中的所有文件和文件夹
        # os.path.isdir(): 检查是否为目录
        # 列表推导式：只保留目录，过滤掉文件
        scenes = [
            f                                           # 文件/目录名
            for f in os.listdir(os.path.join(rootdir, split))  # 遍历分割目录中的所有项
            if os.path.isdir(osp.join(rootdir, split, f))      # 只保留目录
        ]
        
        # ====================================================================
        # 并行处理所有场景 (Process All Scenes in Parallel)
        # ====================================================================
        
        # 使用多进程池并行处理所有场景
        # pool.map(): 将process_scene函数应用到每个参数元组上
        # 为每个场景创建参数元组(rootdir, outdir, split, scene)
        # 多进程处理可以显著提高大数据集的处理速度
        pool.map(process_scene, [(rootdir, outdir, split, scene) for scene in scenes])
    
    # 关闭进程池，不再接受新的任务
    pool.close()
    
    # 等待所有进程完成
    # 这确保主程序在所有场景处理完毕后再继续执行
    pool.join()


# ============================================================================
# 脚本入口点 (Script Entry Point)
# ============================================================================

if __name__ == "__main__":
    """
    脚本的主入口点
    
    当脚本被直接执行时（而不是被导入时），这里的代码会运行
    负责解析命令行参数并启动主处理流程
    """
    # 获取命令行参数解析器
    parser = get_parser()
    
    # 解析命令行参数
    # parse_args(): 解析sys.argv中的命令行参数
    # 返回一个包含所有参数值的Namespace对象
    args = parser.parse_args()
    
    # 调用主函数开始处理
    # args.scannet_dir: 从命令行获取的ScanNet数据集路径
    # args.output_dir: 从命令行获取的输出目录路径
    main(args.scannet_dir, args.output_dir)


# ============================================================================
# 使用示例 (Usage Examples)
# ============================================================================

"""
命令行使用示例：

1. 使用默认路径：
   python preprocess_scannet_explained.py

2. 指定自定义路径：
   python preprocess_scannet_explained.py \
       --scannet_dir /path/to/your/scannet/data \
       --output_dir /path/to/your/output/directory

3. 处理你的数据集（需要修改代码）：
   - 修改内参读取部分以支持JSON格式
   - 修改深度图读取以支持.npy格式
   - 修改位姿读取以支持JSON格式

关键数据流转：
输入 -> 读取 -> 验证 -> 重新格式化 -> 输出
RGB图像: .jpg -> PIL Image -> 调整尺寸 -> .jpg
深度图: .png -> OpenCV -> 保持原格式 -> .png  
内参: .txt -> NumPy数组 -> 验证 -> .npz
位姿: .txt -> 4x4矩阵 -> 验证 -> .npz
""" 