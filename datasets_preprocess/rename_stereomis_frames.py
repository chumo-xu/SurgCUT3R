#!/usr/bin/env python3
"""
Rename StereoMIS frames from 008450-009750 to 000000-001300

This script renames the processed StereoMIS files to match the expected
naming convention for the visualization script (starting from 000000).

Original naming: 008450.jpg, 008451.jpg, ..., 009750.jpg
New naming:      000000.jpg, 000001.jpg, ..., 001300.jpg

Usage:
    python rename_stereomis_frames.py --input_dir /path/to/StereoMIS_processed/StereoMIS_P2_0_1
"""

import os
import argparse
import shutil
from tqdm import tqdm


def rename_files_in_directory(directory, start_frame, end_frame, file_extension):
    """
    Rename files in a directory from original frame numbers to sequential numbers.
    
    Args:
        directory (str): Directory containing the files to rename.
        start_frame (int): Original starting frame number.
        end_frame (int): Original ending frame number.
        file_extension (str): File extension (e.g., '.jpg', '.npy', '.npz').
    
    Returns:
        int: Number of files successfully renamed.
    """
    if not os.path.exists(directory):
        print(f"Warning: Directory does not exist: {directory}")
        return 0
    
    renamed_count = 0
    
    for i, original_frame in enumerate(range(start_frame, end_frame + 1)):
        # Original filename
        original_filename = f"{original_frame:06d}{file_extension}"
        original_path = os.path.join(directory, original_filename)
        
        # New filename (sequential from 000000)
        new_filename = f"{i:06d}{file_extension}"
        new_path = os.path.join(directory, new_filename)
        
        # Check if original file exists
        if os.path.exists(original_path):
            try:
                # Rename the file
                shutil.move(original_path, new_path)
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {original_filename} to {new_filename}: {e}")
        else:
            print(f"Warning: File not found: {original_filename}")
    
    return renamed_count


def rename_stereomis_frames(input_dir, start_frame=8450, end_frame=9750):
    """
    Rename all StereoMIS frame files from original numbering to sequential numbering.
    
    Args:
        input_dir (str): Root directory containing rgb/, depth/, and cam/ subdirectories.
        start_frame (int): Original starting frame number (default: 8450).
        end_frame (int): Original ending frame number (default: 9750).
    """
    print(f"🔄 重命名StereoMIS帧文件...")
    print(f"   原始帧号范围: {start_frame:06d} - {end_frame:06d}")
    print(f"   新帧号范围: 000000 - {end_frame-start_frame:06d}")
    print(f"   总帧数: {end_frame - start_frame + 1}")
    print("=" * 50)
    
    # Define subdirectories and their file extensions
    subdirs_and_extensions = [
        ("rgb", ".jpg"),
        ("depth", ".npy"),
        ("cam", ".npz")
    ]
    
    total_renamed = 0
    
    for subdir, extension in subdirs_and_extensions:
        subdir_path = os.path.join(input_dir, subdir)
        print(f"📁 处理 {subdir}/ 目录 ({extension} 文件)...")
        
        renamed_count = rename_files_in_directory(subdir_path, start_frame, end_frame, extension)
        total_renamed += renamed_count
        
        print(f"   ✅ 成功重命名 {renamed_count} 个文件")
    
    print("=" * 50)
    print(f"🎉 重命名完成！总共处理了 {total_renamed} 个文件")
    
    # Verify the results
    print("\n🔍 验证重命名结果...")
    expected_files_per_dir = end_frame - start_frame + 1
    
    for subdir, extension in subdirs_and_extensions:
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.exists(subdir_path):
            # Count files with new naming pattern
            new_files = [f for f in os.listdir(subdir_path) 
                        if f.endswith(extension) and f.startswith('0')]
            print(f"   {subdir}/: {len(new_files)} 个文件 (期望: {expected_files_per_dir})")
            
            # Check if we have the expected range
            if len(new_files) == expected_files_per_dir:
                # Verify first and last files exist
                first_file = f"000000{extension}"
                last_file = f"{expected_files_per_dir-1:06d}{extension}"
                
                first_exists = os.path.exists(os.path.join(subdir_path, first_file))
                last_exists = os.path.exists(os.path.join(subdir_path, last_file))
                
                if first_exists and last_exists:
                    print(f"   ✅ {subdir}/ 重命名成功！")
                else:
                    print(f"   ⚠️  {subdir}/ 可能存在问题")
            else:
                print(f"   ⚠️  {subdir}/ 文件数量不匹配")


def main():
    parser = argparse.ArgumentParser(
        description="重命名StereoMIS帧文件，从原始帧号改为从000000开始的连续帧号"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含rgb/, depth/, cam/子目录的StereoMIS数据目录路径",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=8450,
        help="原始起始帧号 (默认: 8450)",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=9750,
        help="原始结束帧号 (默认: 9750)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅显示将要执行的操作，不实际重命名文件",
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ 错误: 输入目录不存在: {args.input_dir}")
        return
    
    # Check if required subdirectories exist
    required_subdirs = ["rgb", "depth", "cam"]
    missing_subdirs = []
    for subdir in required_subdirs:
        subdir_path = os.path.join(args.input_dir, subdir)
        if not os.path.exists(subdir_path):
            missing_subdirs.append(subdir)
    
    if missing_subdirs:
        print(f"❌ 错误: 缺少必需的子目录: {missing_subdirs}")
        return
    
    if args.dry_run:
        print("🔍 DRY RUN 模式 - 仅显示操作，不实际执行")
        print(f"将要重命名的文件:")
        print(f"  {args.input_dir}/rgb/{args.start_frame:06d}.jpg -> 000000.jpg")
        print(f"  {args.input_dir}/rgb/{args.start_frame+1:06d}.jpg -> 000001.jpg")
        print(f"  ...")
        print(f"  {args.input_dir}/rgb/{args.end_frame:06d}.jpg -> {args.end_frame-args.start_frame:06d}.jpg")
        print(f"总共 {args.end_frame - args.start_frame + 1} 个文件 × 3 个目录 = {(args.end_frame - args.start_frame + 1) * 3} 个文件")
        return
    
    # Confirm before proceeding
    print(f"⚠️  即将重命名 {args.input_dir} 中的文件")
    print(f"   原始帧号: {args.start_frame} - {args.end_frame}")
    print(f"   新帧号: 0 - {args.end_frame - args.start_frame}")
    
    confirm = input("确认继续? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 操作已取消")
        return
    
    # Perform the renaming
    rename_stereomis_frames(args.input_dir, args.start_frame, args.end_frame)


if __name__ == "__main__":
    main()
