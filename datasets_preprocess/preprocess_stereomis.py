#!/usr/bin/env python3
"""
Preprocess Script for StereoMIS Dataset (Batch Processing)

This script processes the complete StereoMIS dataset by:
  - Converting PNG images to JPG format (no resizing, keeping 1024x1280).
  - Copying depth NPY files.
  - Reading camera intrinsics from K.txt file (no scaling).
  - Reading camera poses from groundtruth.txt file and converting quaternion+translation to 4x4 matrices.
  - Processing all P folders and their image sequences automatically.
  - Saving all data in the same format as Point Odyssey dataset.

The StereoMIS dataset structure:
  - Images: /path/to/Endo-4DGS/data/StereoMIS_0_0_1/StereoMIS_0_0_1/P2_X/images_Y/XXXXXX.png
  - Depths: /path/to/FoundationStereo/batch_depth_outputs_full/P2_X/pair_Y/XXXXXX.npy
  - Poses: /path/to/P2_X/groundtruth.txt (each line: timestamp tx ty tz qx qy qz qw)
  - Intrinsics: /path/to/P2_X/K.txt (3x3 matrix + baseline)

Output structure (following Point Odyssey format):
  - StereoMIS_P2_X_Y/rgb/XXXXXX.jpg
  - StereoMIS_P2_X_Y/depth/XXXXXX.npy
  - StereoMIS_P2_X_Y/cam/XXXXXX.npz

Usage:
    python preprocess_stereomis.py --input_dir /path/to/dataset_root --output_dir /path/to/output_dataset
"""

import os
import argparse
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import glob


def discover_sequences(dataset_root):
    """
    Discover all P folders and their image sequences in the dataset.

    Args:
        dataset_root (str): Root directory of the dataset.

    Returns:
        list: List of tuples (p_folder, image_sequence_num) for all found sequences.
    """
    sequences = []

    # Path to the main data directory
    data_dir = os.path.join(dataset_root, "hy-tmp", "Endo-4DGS", "data", "StereoMIS_0_0_1", "StereoMIS_0_0_1")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return sequences

    # Find all P folders
    p_folders = [d for d in os.listdir(data_dir) if d.startswith('P') and os.path.isdir(os.path.join(data_dir, d))]
    p_folders.sort()

    for p_folder in p_folders:
        p_path = os.path.join(data_dir, p_folder)

        # Find all images_X folders in this P folder (exclude right eye images)
        image_folders = [d for d in os.listdir(p_path)
                        if d.startswith('images_') and not d.startswith('images_right_')
                        and os.path.isdir(os.path.join(p_path, d))]

        for image_folder in image_folders:
            # Extract sequence number from images_X
            try:
                seq_num = int(image_folder.split('_')[1])
                sequences.append((p_folder, seq_num))
            except (IndexError, ValueError):
                print(f"Warning: Could not parse sequence number from {image_folder}")
                continue

    print(f"Discovered {len(sequences)} sequences: {sequences}")
    return sequences


def load_intrinsics(intrinsics_file):
    """
    Load camera intrinsics from K.txt file (no scaling applied).

    Args:
        intrinsics_file (str): Path to the K.txt file containing intrinsics.

    Returns:
        np.ndarray: 3x3 intrinsics matrix.
    """
    with open(intrinsics_file, 'r') as f:
        lines = f.readlines()

    # Parse the first line as 3x3 intrinsics matrix (flattened)
    intrinsics_flat = list(map(float, lines[0].strip().split()))

    # Reshape to 3x3 matrix
    intrinsics = np.array(intrinsics_flat, dtype=np.float32).reshape(3, 3)

    return intrinsics


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        qx, qy, qz, qw (float): Quaternion components.
        
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    # Normalize quaternion
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float32)
    
    return R


def pose_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    """
    Convert translation and quaternion to 4x4 pose matrix.
    
    Args:
        tx, ty, tz (float): Translation components.
        qx, qy, qz, qw (float): Quaternion components.
        
    Returns:
        np.ndarray: 4x4 pose matrix (camera-to-world).
    """
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    
    # Create 4x4 pose matrix
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    
    return pose


def load_poses(pose_file, start_frame, end_frame):
    """
    Load camera poses from groundtruth.txt file for specified frame range.

    Args:
        pose_file (str): Path to the groundtruth.txt file.
        start_frame (int): Starting frame number.
        end_frame (int): Ending frame number.

    Returns:
        dict: Dictionary mapping frame numbers to 4x4 pose matrices.
    """
    poses = {}

    with open(pose_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 8:
            print(f"Warning: Invalid pose line: {line}")
            continue

        try:
            # Parse pose data: timestamp tx ty tz qx qy qz qw
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            timestamp = int(timestamp)

            # Check if this timestamp is in our range
            if start_frame <= timestamp <= end_frame:
                # Convert to 4x4 matrix (assuming c2w format as requested)
                pose = pose_to_matrix(tx, ty, tz, qx, qy, qz, qw)
                poses[timestamp] = pose

        except ValueError as e:
            print(f"Warning: Could not parse pose line: {line}, error: {e}")
            continue

    return poses


def get_sequence_frame_range(img_dir):
    """
    Automatically detect the frame range for a sequence by scanning image files.

    Args:
        img_dir (str): Path to the images directory.

    Returns:
        tuple: (start_frame, end_frame, total_frames) or (None, None, 0) if no frames found.
    """
    if not os.path.exists(img_dir):
        return None, None, 0

    png_files = glob.glob(os.path.join(img_dir, '*.png'))
    if not png_files:
        return None, None, 0

    frame_nums = []
    for f in png_files:
        try:
            frame_num = int(os.path.basename(f).split('.')[0])
            frame_nums.append(frame_num)
        except:
            continue

    if not frame_nums:
        return None, None, 0

    frame_nums.sort()
    return frame_nums[0], frame_nums[-1], len(frame_nums)


def process_single_sequence(input_dir, output_dir, p_folder, seq_num):
    """
    Process a single StereoMIS sequence with automatic frame range detection.

    Args:
        input_dir (str): Root input directory containing StereoMIS data.
        output_dir (str): Output directory where processed files will be saved.
        p_folder (str): P folder name (e.g., 'P2_0').
        seq_num (int): Sequence number (e.g., 1 for images_1).
    """
    print(f"\nProcessing sequence: {p_folder}/images_{seq_num}")

    # Define input paths
    data_root = os.path.join(input_dir, "hy-tmp", "Endo-4DGS", "data", "StereoMIS_0_0_1", "StereoMIS_0_0_1")
    depth_root = os.path.join(input_dir, "hy-tmp", "FoundationStereo", "batch_depth_outputs_full")

    img_dir = os.path.join(data_root, p_folder, f"images_{seq_num}")
    depth_dir = os.path.join(depth_root, p_folder, f"pair_{seq_num}")
    pose_file = os.path.join(data_root, p_folder, "groundtruth.txt")
    intrinsics_file = os.path.join(data_root, p_folder, "K.txt")

    # Check if all required paths exist
    if not all(os.path.exists(path) for path in [img_dir, depth_dir, pose_file, intrinsics_file]):
        missing_paths = [path for path in [img_dir, depth_dir, pose_file, intrinsics_file] if not os.path.exists(path)]
        print(f"Error: Missing required paths for {p_folder}/images_{seq_num}: {missing_paths}")
        return False

    # Auto-detect frame range
    start_frame, end_frame, total_frames = get_sequence_frame_range(img_dir)
    if start_frame is None:
        print(f"Error: No frames found in {img_dir}")
        return False

    print(f"Detected frame range: {start_frame}-{end_frame} ({total_frames} frames)")

    # Create output directories
    out_seq_dir = os.path.join(output_dir, f"StereoMIS_{p_folder}_{seq_num}")
    out_rgb_dir = os.path.join(out_seq_dir, "rgb")
    out_depth_dir = os.path.join(out_seq_dir, "depth")
    out_cam_dir = os.path.join(out_seq_dir, "cam")

    os.makedirs(out_rgb_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_cam_dir, exist_ok=True)

    # Load camera intrinsics (fixed for the entire sequence)
    print("Loading camera intrinsics...")
    print("Note: Using original 1024x1280 resolution, no scaling applied")
    intrinsics = load_intrinsics(intrinsics_file)
    print(f"Intrinsics matrix:\n{intrinsics}")

    # Load camera poses for the frame range
    print(f"Loading camera poses for frames {start_frame} to {end_frame}...")
    poses = load_poses(pose_file, start_frame, end_frame)
    print(f"Loaded {len(poses)} poses")

    # Process each frame
    num_frames = end_frame - start_frame + 1
    print(f"Processing {num_frames} frames...")

    processed_count = 0
    for frame_id in tqdm(range(start_frame, end_frame + 1), desc=f"Processing {p_folder}/images_{seq_num}"):
        frame_str = f"{frame_id:06d}"

        # Define file paths
        img_path = os.path.join(img_dir, f"{frame_str}.png")
        depth_path = os.path.join(depth_dir, f"{frame_str}.npy")

        # Check if files exist
        if not os.path.exists(img_path):
            continue  # Skip missing images silently
        if not os.path.exists(depth_path):
            continue  # Skip missing depth files silently
        if frame_id not in poses:
            continue  # Skip frames without poses silently

        try:
            # Process RGB image (PNG -> JPG, no resizing)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            out_img_path = os.path.join(out_rgb_dir, f"{frame_str}.jpg")
            img.save(out_img_path, 'JPEG', quality=95)

            # Copy depth data
            depth = np.load(depth_path).astype(np.float32)
            out_depth_path = os.path.join(out_depth_dir, f"{frame_str}.npy")
            np.save(out_depth_path, depth)

            # Get pose for this frame
            pose = poses[frame_id]

            # Validate pose matrix
            if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
                print(f"Warning: Invalid pose for frame {frame_id}. Skipping.")
                continue

            # Save camera parameters (intrinsics + pose)
            out_cam_path = os.path.join(out_cam_dir, f"{frame_str}.npz")
            np.savez(out_cam_path, intrinsics=intrinsics, pose=pose)

            processed_count += 1

        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            continue

    print(f"Successfully processed {processed_count} out of {num_frames} frames for {p_folder}/images_{seq_num}")
    return True


def process_all_sequences(input_dir, output_dir):
    """
    Process all discovered sequences in the StereoMIS dataset with automatic frame range detection.

    Args:
        input_dir (str): Root input directory containing StereoMIS data.
        output_dir (str): Output directory where processed files will be saved.
    """
    print("Discovering sequences in the dataset...")
    sequences = discover_sequences(input_dir)

    if not sequences:
        print("No sequences found in the dataset!")
        return

    print(f"Found {len(sequences)} sequences to process")

    successful_count = 0
    failed_count = 0

    for p_folder, seq_num in sequences:
        try:
            success = process_single_sequence(input_dir, output_dir, p_folder, seq_num)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"Error processing {p_folder}/images_{seq_num}: {e}")
            failed_count += 1

    print(f"\nProcessing completed!")
    print(f"Successfully processed: {successful_count} sequences")
    print(f"Failed to process: {failed_count} sequences")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess complete StereoMIS dataset by processing all P folders and image sequences with automatic frame range detection."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/hy-tmp/hy-tmp/CUT3R/dataset",
        help="Path to the input dataset root directory (default: /hy-tmp/hy-tmp/CUT3R/dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/hy-tmp/hy-tmp/CUT3R/dataset/processed_stereomis",
        help="Path to the root output directory (default: /hy-tmp/hy-tmp/CUT3R/dataset/processed_stereomis).",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all sequences in the StereoMIS dataset
    process_all_sequences(args.input_dir, args.output_dir)

    print("StereoMIS dataset preprocessing completed!")


if __name__ == "__main__":
    main()
