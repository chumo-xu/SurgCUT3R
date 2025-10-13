#!/usr/bin/env python3
"""
Preprocess Script for C3VD Dataset

This script processes the C3VD dataset by:
  - Converting RGB images from PNG to JPG format.
  - Converting depth TIFF files to real depth values in millimeters.
  - Reading camera poses from pose.txt file.
  - Using estimated camera intrinsics for endoscopic cameras.
  - Saving all data in the same format as SCARED dataset.

The C3VD dataset structure:
  - {0-275}_color.png: RGB images
  - {0000-0275}_depth.tiff: Depth images (0-100mm encoded as 16-bit)
  - pose.txt: Camera poses (camera-to-world transformation matrices)

Output structure (following SCARED format):
  - C3VD_{sequence_name}/rgb/{frame_id}.jpg
  - C3VD_{sequence_name}/depth/{frame_id}.npy  
  - C3VD_{sequence_name}/cam/{frame_id}.npz

Usage:
    python preprocess_c3vd.py --input_dir /path/to/cecum_t1_a --output_dir /path/to/output_dataset --sequence_name cecum_t1_a
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def get_camera_intrinsics(width=1350, height=1080):
    """
    Get estimated camera intrinsics for endoscopic camera.

    Args:
        width (int): Image width
        height (int): Image height

    Returns:
        np.ndarray: 3x3 intrinsics matrix
    """
    import math

    # 内窥镜相机通常有较大的视场角
    # 假设水平FOV = 90度（比较常见的内窥镜参数）
    fov_horizontal_deg = 90.0
    fov_horizontal_rad = math.radians(fov_horizontal_deg)

    # 基于FOV计算焦距：fx = width / (2 * tan(FOV/2))
    fx = width / (2.0 * math.tan(fov_horizontal_rad / 2.0))
    fy = fx  # 假设像素是正方形

    # 主点通常在图像中心
    cx = width / 2.0
    cy = height / 2.0

    print(f"估算的相机内参 (FOV={fov_horizontal_deg}°):")
    print(f"  fx = fy = {fx:.1f} (原来是1000.0)")
    print(f"  cx = {cx:.1f}, cy = {cy:.1f}")

    intrinsics = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return intrinsics


def load_poses(pose_file):
    """
    Load camera poses from pose.txt file.
    
    Args:
        pose_file (str): Path to pose.txt file
        
    Returns:
        list: List of 4x4 pose matrices
    """
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            # Parse 16 values (4x4 matrix in row-major order)
            values = [float(x) for x in line.split(',')]
            if len(values) != 16:
                print(f"Warning: Invalid pose line with {len(values)} values: {line}")
                continue
                
            # Reshape to 4x4 matrix (按行存储)
            pose = np.array(values, dtype=np.float32).reshape(4, 4)

            # C3VD的位姿矩阵格式需要转置！
            # 原始数据是按行存储：[R11,R12,R13,0, R21,R22,R23,0, R31,R32,R33,0, tx,ty,tz,1]
            # 但reshape(4,4)会按行填充，导致位移向量在最后一行而不是最后一列
            # 需要转置使其变为标准的cam2world格式：位移在最后一列
            pose = pose.T

            # 将位移向量从毫米转换为米
            pose[:3, 3] = pose[:3, 3] / 1000.0

            # 验证转置后的位姿矩阵
            if not np.isfinite(pose).all():
                print(f"Warning: Invalid pose matrix with non-finite values")
                continue

            # 打印第一个位姿矩阵用于调试
            if len(poses) == 0:
                print(f"第一个位姿矩阵 (转置后，位移已转换为米):")
                print(pose)
                print(f"位移向量 (米): {pose[:3, 3]}")
                print(f"旋转矩阵:")
                print(pose[:3, :3])
                
            poses.append(pose)
    
    return poses


def convert_depth(depth_uint16):
    """
    Convert depth from uint16 encoding to real depth in millimeters.
    
    Depth encoding: 0-100mm linearly scaled to 0-65535
    
    Args:
        depth_uint16 (np.ndarray): Encoded depth image
        
    Returns:
        np.ndarray: Real depth in millimeters
    """
    # Convert to float32 and scale to real depth
    depth_mm = depth_uint16.astype(np.float32) * (100.0 / 65535.0)
    return depth_mm


def process_sequence(input_dir, output_dir, sequence_name):
    """
    Process a single C3VD sequence.
    
    Args:
        input_dir (str): Input directory containing the sequence data
        output_dir (str): Output directory for processed data
        sequence_name (str): Name of the sequence (e.g., 'cecum_t1_a')
    """
    print(f"Processing C3VD sequence: {sequence_name}")
    
    # Check input files
    pose_file = os.path.join(input_dir, "pose.txt")
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")
    
    # Load poses
    print("Loading camera poses...")
    poses = load_poses(pose_file)
    print(f"Loaded {len(poses)} poses")
    
    # Find all RGB and depth files dynamically
    rgb_files = []
    depth_files = []

    # Scan directory for all available files
    all_files = os.listdir(input_dir)
    rgb_pattern = [f for f in all_files if f.endswith('_color.png')]
    depth_pattern = [f for f in all_files if f.endswith('_depth.tiff')]

    # Extract frame numbers and sort
    rgb_frames = []
    for f in rgb_pattern:
        try:
            frame_num = int(f.split('_')[0])
            rgb_frames.append(frame_num)
        except ValueError:
            continue

    depth_frames = []
    for f in depth_pattern:
        try:
            frame_num = int(f.split('_')[0])
            depth_frames.append(frame_num)
        except ValueError:
            continue

    # Find common frames
    common_frames = sorted(set(rgb_frames) & set(depth_frames))
    print(f"Found {len(rgb_frames)} RGB files, {len(depth_frames)} depth files")
    print(f"Common frames: {len(common_frames)} (from {min(common_frames) if common_frames else 'N/A'} to {max(common_frames) if common_frames else 'N/A'})")

    # Build file lists for common frames
    # Try all possible naming conventions combinations
    for frame_num in common_frames:
        rgb_file = None
        depth_file = None

        # Try all combinations of padded/unpadded for RGB and depth files
        rgb_candidates = [
            os.path.join(input_dir, f"{frame_num:04d}_color.png"),  # padded RGB
            os.path.join(input_dir, f"{frame_num}_color.png")       # unpadded RGB
        ]

        depth_candidates = [
            os.path.join(input_dir, f"{frame_num:04d}_depth.tiff"), # padded depth
            os.path.join(input_dir, f"{frame_num}_depth.tiff")      # unpadded depth
        ]

        # Find existing RGB file
        for rgb_candidate in rgb_candidates:
            if os.path.exists(rgb_candidate):
                rgb_file = rgb_candidate
                break

        # Find existing depth file
        for depth_candidate in depth_candidates:
            if os.path.exists(depth_candidate):
                depth_file = depth_candidate
                break

        # Check if both files were found
        if rgb_file and depth_file:
            rgb_files.append((frame_num, rgb_file))
            depth_files.append((frame_num, depth_file))
        else:
            missing = []
            if not rgb_file:
                missing.append("RGB")
            if not depth_file:
                missing.append("depth")
            print(f"Warning: Could not find {' and '.join(missing)} file(s) for frame {frame_num}")
            continue

    print(f"Will process {len(rgb_files)} matched RGB-depth pairs")
    
    # Verify and adjust to minimum count
    min_count = min(len(rgb_files), len(depth_files), len(poses))
    print(f"Data summary: {len(rgb_files)} RGB files, {len(depth_files)} depth files, {len(poses)} poses")
    print(f"Will process {min_count} frames (minimum of all three)")

    if min_count == 0:
        raise ValueError("No valid frames found to process")

    # Trim to minimum count
    rgb_files = rgb_files[:min_count]
    depth_files = depth_files[:min_count]
    poses = poses[:min_count]
    
    # Create output directories
    # output_dir should be the final sequence directory (e.g., C3VD_cecum_t1_a)
    out_rgb_dir = os.path.join(output_dir, "rgb")
    out_depth_dir = os.path.join(output_dir, "depth")
    out_cam_dir = os.path.join(output_dir, "cam")
    
    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)
    
    # Get camera intrinsics (will be determined from first image)
    intrinsics = None
    
    # Process each frame
    for idx, ((frame_id, rgb_file), (_, depth_file)) in enumerate(tqdm(zip(rgb_files, depth_files), 
                                                                        desc=f"Processing {sequence_name}", 
                                                                        total=len(rgb_files))):
        
        # Read RGB image
        rgb = cv2.imread(rgb_file)
        if rgb is None:
            print(f"Warning: Could not read RGB image {rgb_file}. Skipping frame {frame_id}.")
            continue
        
        # Set intrinsics based on first image
        if intrinsics is None:
            height, width = rgb.shape[:2]
            intrinsics = get_camera_intrinsics(width, height)
            print(f"Using intrinsics for {width}x{height} images:")
            print(intrinsics)
        
        # Read depth image
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Could not read depth image {depth_file}. Skipping frame {frame_id}.")
            continue
        
        # Convert depth to real values in millimeters, then convert to meters
        depth_mm = convert_depth(depth)
        depth_m = depth_mm / 1000.0  # Convert mm to m
        
        # Get corresponding pose
        if idx >= len(poses):
            print(f"Warning: No pose available for frame {frame_id}. Skipping.")
            continue
        pose = poses[idx]
        
        # Save RGB image as JPG (6-digit format for compatibility)
        out_rgb_path = os.path.join(out_rgb_dir, f"{frame_id:06d}.jpg")
        cv2.imwrite(out_rgb_path, rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save depth as NPY in meters (6-digit format for compatibility)
        out_depth_path = os.path.join(out_depth_dir, f"{frame_id:06d}.npy")
        np.save(out_depth_path, depth_m)

        # Save camera parameters (6-digit format for compatibility)
        out_cam_path = os.path.join(out_cam_dir, f"{frame_id:06d}.npz")
        np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)
    
    print(f"Successfully processed {len(rgb_files)} frames for sequence {sequence_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess C3VD dataset by processing images, depth maps, and camera parameters."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input sequence directory (e.g., /path/to/cecum_t1_a).",
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required=True,
        help="Path to the root output directory where processed data will be stored.",
    )
    parser.add_argument(
        "--sequence_name",
        type=str,
        required=True,
        help="Name of the sequence (e.g., 'cecum_t1_a').",
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process the sequence
    process_sequence(args.input_dir, args.output_dir, args.sequence_name)
    
    print("C3VD dataset preprocessing completed!")


if __name__ == "__main__":
    main()
