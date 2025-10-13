#!/usr/bin/env python3
"""
Preprocess Script for SCARED Dataset

This script processes the SCARED dataset by:
  - Copying RGB images from left_rectified folders.
  - Converting depth NPY files to the output format.
  - Reading camera intrinsics from model_input_params.json.
  - Reading camera poses from pose JSON files and converting w2c to c2w.
  - Saving all data in the same format as Point Odyssey dataset.

The SCARED dataset structure (new format):
  - hy-tmp/output/training_data_left_only/dataset{N}/keyframe{M}/left_rectified/XXXXXX.png
  - hy-tmp/output/training_data_left_only/dataset{N}/keyframe{M}/model_input_params.json
  - output_full_batch_depth/dataset{N}/keyframe{M}/depth_XXXXXX.npy
  - scared12367pose/frame_data{N}-{M}/frame_dataXXXXXX.json

Output structure (following Point Odyssey format):
  - dataset{N}_keyframe{M}/rgb/XXXXXX.jpg
  - dataset{N}_keyframe{M}/depth/XXXXXX.npy
  - dataset{N}_keyframe{M}/cam/XXXXXX.npz

Usage:
    python preprocess_scared.py --input_dir /path/to/FoundationStereo --output_dir /path/to/output_dataset
"""

import os
import argparse
import shutil
import numpy as np
import json
import cv2
from tqdm import tqdm


def load_intrinsics(intrinsics_file):
    """
    Load camera intrinsics from model_input_params.json file.
    
    Args:
        intrinsics_file (str): Path to the JSON file containing intrinsics.
        
    Returns:
        np.ndarray: 3x3 intrinsics matrix.
    """
    with open(intrinsics_file, 'r') as f:
        params = json.load(f)
    
    # 构建3x3内参矩阵
    fx, fy = params['fx'], params['fy']
    cx, cy = params['cx'], params['cy']
    
    intrinsics = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    return intrinsics


def load_pose(pose_file):
    """
    Load camera pose from frame_dataXXXXXX.json file and convert from w2c to c2w.
    Also converts translation units from mm to m to match depth units.

    Args:
        pose_file (str): Path to the JSON file containing pose.

    Returns:
        np.ndarray: 4x4 camera pose matrix (c2w format) with translation in meters.
    """
    with open(pose_file, 'r') as f:
        data = json.load(f)

    # 提取camera-pose矩阵 (w2c格式)
    pose_w2c = np.array(data['camera-pose'], dtype=np.float32)

    # 转换为c2w格式 (取逆)
    pose_c2w = np.linalg.inv(pose_w2c)

    # 将位移向量从mm转换为m (深度单位是m，位姿位移单位是mm)
    pose_c2w[:3, 3] = pose_c2w[:3, 3] / 1000.0

    return pose_c2w


def process_keyframe(dataset_num, keyframe_num, input_dir, output_dir):
    """
    Process a single keyframe:
      - Dynamically detects the number of frames in the keyframe.
      - Loads intrinsics, poses, RGB images, and depth maps.
      - Saves the results in Point Odyssey format.

    Args:
        dataset_num (int): Dataset number (1, 2, 3, 6, 7).
        keyframe_num (int): Keyframe number.
        input_dir (str): Root input directory containing data_all.
        output_dir (str): Output directory where processed files will be saved.
    """
    # 定义输入路径
    depth_dir = os.path.join(input_dir, "output_full_batch_depth", f"dataset{dataset_num}", f"keyframe{keyframe_num}")
    img_dir = os.path.join(input_dir, "hy-tmp", "output", "training_data_left_only", f"dataset{dataset_num}", f"keyframe{keyframe_num}", "left_rectified")
    intrinsics_file = os.path.join(input_dir, "hy-tmp", "output", "training_data_left_only", f"dataset{dataset_num}", f"keyframe{keyframe_num}", "model_input_params.json")
    pose_dir = os.path.join(input_dir, "scared12367pose", f"frame_data{dataset_num}-{keyframe_num}")
    
    # 检查必要文件/文件夹是否存在
    if not all(os.path.exists(path) for path in [depth_dir, img_dir, intrinsics_file, pose_dir]):
        print(f"Warning: Missing data for dataset{dataset_num}/keyframe{keyframe_num}. Skipping.")
        return
    
    # 动态获取深度文件列表
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith('depth_') and f.endswith('.npy')])
    if not depth_files:
        print(f"Warning: No depth files found for dataset{dataset_num}/keyframe{keyframe_num}. Skipping.")
        return
    
    # 动态获取图像文件列表
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if not img_files:
        print(f"Warning: No image files found for dataset{dataset_num}/keyframe{keyframe_num}. Skipping.")
        return
    
    # 动态获取外参文件列表
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.startswith('frame_data') and f.endswith('.json')])
    if not pose_files:
        print(f"Warning: No pose files found for dataset{dataset_num}/keyframe{keyframe_num}. Skipping.")
        return
    
    # 验证文件数量是否匹配
    if not (len(depth_files) == len(img_files) == len(pose_files)):
        print(f"Warning: File count mismatch for dataset{dataset_num}/keyframe{keyframe_num}: "
              f"{len(depth_files)} depths, {len(img_files)} images, {len(pose_files)} poses. Skipping.")
        return
    
    num_frames = len(depth_files)
    print(f"Processing dataset{dataset_num}/keyframe{keyframe_num} with {num_frames} frames...")
    
    # 创建输出目录
    out_seq_dir = os.path.join(output_dir, f"dataset{dataset_num}_keyframe{keyframe_num}")
    out_rgb_dir = os.path.join(out_seq_dir, "rgb")
    out_depth_dir = os.path.join(out_seq_dir, "depth") 
    out_cam_dir = os.path.join(out_seq_dir, "cam")
    
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)
    
    # 加载相机内参 (对整个keyframe是固定的)
    intrinsics = load_intrinsics(intrinsics_file)
    
    # 处理每一帧
    for i in tqdm(range(num_frames), desc=f"Processing dataset{dataset_num}_keyframe{keyframe_num}", leave=False):
        # 提取帧号
        depth_file = depth_files[i]
        img_file = img_files[i] 
        pose_file = pose_files[i]
        
        # 从文件名提取帧号并验证一致性
        depth_frame_id = depth_file.split('_')[1].split('.')[0]  # depth_XXXXXX.npy -> XXXXXX
        img_frame_id = img_file.split('.')[0]  # XXXXXX.png -> XXXXXX
        pose_frame_id = pose_file.split('frame_data')[1].split('.')[0]  # frame_dataXXXXXX.json -> XXXXXX
        
        if not (depth_frame_id == img_frame_id == pose_frame_id):
            print(f"Warning: Frame ID mismatch at index {i}: depth={depth_frame_id}, img={img_frame_id}, pose={pose_frame_id}. Skipping frame.")
            continue
        
        frame_id = depth_frame_id
        
        # 读取和处理数据
        # 1. 读取并复制RGB图像 (转换为JPG格式)
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping frame {frame_id}.")
            continue
        out_img_path = os.path.join(out_rgb_dir, frame_id + ".jpg")
        cv2.imwrite(out_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 2. 读取并保存深度数据
        depth_path = os.path.join(depth_dir, depth_file)
        try:
            depth = np.load(depth_path).astype(np.float32)
        except Exception as e:
            print(f"Warning: Could not read depth {depth_path}: {e}. Skipping frame {frame_id}.")
            continue
        out_depth_path = os.path.join(out_depth_dir, frame_id + ".npy")
        np.save(out_depth_path, depth)
        
        # 3. 读取外参并保存相机参数
        pose_path = os.path.join(pose_dir, pose_file)
        try:
            pose = load_pose(pose_path)
        except Exception as e:
            print(f"Warning: Could not read pose {pose_path}: {e}. Skipping frame {frame_id}.")
            continue
        
        # 验证pose矩阵有效性
        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            print(f"Warning: Invalid pose for frame {frame_id}. Skipping.")
            continue
        
        # 保存相机参数 (内参 + 外参)
        out_cam_path = os.path.join(out_cam_dir, frame_id + ".npz")
        np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)


def process_dataset(input_dir, output_dir):
    """
    Process the entire SCARED dataset.

    Args:
        input_dir (str): Root input directory containing GTdepth, left_images, pose folders.
        output_dir (str): Output directory where processed data will be stored.
    """
    # SCARED数据集包含的dataset编号
    dataset_numbers = [1, 2, 3, 6, 7]
    
    for dataset_num in dataset_numbers:
        # 确定该dataset下有哪些keyframe
        dataset_dir = os.path.join(input_dir, "output_full_batch_depth", f"dataset{dataset_num}")
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset{dataset_num} directory does not exist. Skipping.")
            continue
        
        keyframes = sorted([d for d in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('keyframe')])
        
        if not keyframes:
            print(f"Warning: No keyframes found for dataset{dataset_num}. Skipping.")
            continue
        
        print(f"Found {len(keyframes)} keyframes for dataset{dataset_num}: {keyframes}")
        
        for keyframe_dir in keyframes:
            # 提取keyframe编号
            keyframe_num = int(keyframe_dir.replace('keyframe', ''))
            process_keyframe(dataset_num, keyframe_num, input_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SCARED dataset by processing images, depth maps, and camera parameters."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input FoundationStereo directory containing output_full_batch_depth, hy-tmp/output/training_data_left_only, scared12367pose folders.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Path to the root output directory where processed data will be stored.",
    )
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理整个数据集
    process_dataset(args.input_dir, args.output_dir)
    
    print("SCARED dataset preprocessing completed!")


if __name__ == "__main__":
    main() 