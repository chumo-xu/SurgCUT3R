#!/usr/bin/env python3
# =============================================================================
# Point Odyssey数据集预处理代码详细解释版本
# 原始文件：preprocess_point_odyssey.py
# =============================================================================

"""
Point Odyssey数据集预处理脚本详细解释

这个脚本用于处理Point Odyssey数据集，主要功能包括：
  - 复制RGB图像文件
  - 将16位深度图像转换为归一化的float32深度图
  - 反转相机外参矩阵以获得位姿
  - 将内参和计算得到的位姿保存到结构化的输出目录中

数据集预期的结构：
- 每个分割（如train, test, val）有子目录
- 每个分割包含多个序列目录
- 每个序列目录必须包含以下内容：
  - 'rgbs' 文件夹：包含.jpg图像
  - 'depths' 文件夹：包含.png深度图像
  - 'anno.npz' 文件：包含'intrinsics'和'extrinsics'数组

使用方法：
    python preprocess_point_odyssey.py --input_dir /path/to/input_dataset --output_dir /path/to/output_dataset
"""

# =============================================================================
# 库导入部分 (Library Imports)
# =============================================================================

import os                   # 操作系统接口模块，用于路径操作和目录管理
import argparse            # 命令行参数解析库
import shutil              # 高级文件操作库，用于复制文件
import numpy as np         # 数值计算库，用于处理数组和矩阵运算
import cv2                 # OpenCV计算机视觉库，用于读取深度图像
from tqdm import tqdm      # 进度条显示库，用于显示处理进度


def process_sequence(seq_dir, out_seq_dir):
    """
    处理单个序列的所有数据
    
    这个函数是处理单个序列的核心函数，执行以下步骤：
      - 验证必需的文件夹/文件是否存在
      - 加载相机标注数据
      - 处理每一帧：复制RGB图像，处理深度图，计算相机位姿，保存结果

    参数说明：
        seq_dir (str): 序列目录路径（应包含'rgbs', 'depths', 和'anno.npz'）
        out_seq_dir (str): 输出目录路径，处理后的文件将保存在这里
    """
    
    # =========================================================================
    # 定义输入文件和目录路径 (Define Input File and Directory Paths)
    # =========================================================================
    
    # 构建RGB图像目录路径
    # os.path.join(): 跨平台的路径连接函数，自动处理不同操作系统的路径分隔符
    img_dir = os.path.join(seq_dir, "rgbs")
    
    # 构建深度图目录路径
    depth_dir = os.path.join(seq_dir, "depths")
    
    # 构建相机标注文件路径
    # Point Odyssey数据集使用NPZ格式存储相机内参和外参
    cam_file = os.path.join(seq_dir, "anno.npz")

    # =========================================================================
    # 验证输入数据完整性 (Verify Input Data Integrity)
    # =========================================================================
    
    # 检查所有必需的文件/文件夹是否存在
    # os.path.exists(): 检查路径是否存在
    # not (...and...and...): 如果任何一个路径不存在，条件为True
    if not (
        os.path.exists(img_dir)           # RGB图像目录必须存在
        and os.path.exists(depth_dir)     # 深度图目录必须存在
        and os.path.exists(cam_file)      # 相机标注文件必须存在
    ):
        # 如果任何必需文件缺失，抛出文件未找到异常
        raise FileNotFoundError(f"Missing required data in {seq_dir}")

    # =========================================================================
    # 创建输出目录结构 (Create Output Directory Structure)
    # =========================================================================
    
    # 为图像、深度图和相机参数创建输出子目录
    out_img_dir = os.path.join(out_seq_dir, "rgb")      # RGB图像输出目录
    out_depth_dir = os.path.join(out_seq_dir, "depth")  # 深度图输出目录
    out_cam_dir = os.path.join(out_seq_dir, "cam")      # 相机参数输出目录
    
    # os.makedirs(): 递归创建目录
    # exist_ok=True: 如果目录已存在，不会抛出异常
    os.makedirs(out_img_dir, exist_ok=True)     # 创建RGB输出目录
    os.makedirs(out_depth_dir, exist_ok=True)   # 创建深度输出目录
    os.makedirs(out_cam_dir, exist_ok=True)     # 创建相机参数输出目录

    # =========================================================================
    # 加载相机标注数据 (Load Camera Annotation Data)
    # =========================================================================
    
    # 使用NumPy加载NPZ格式的标注文件
    # np.load(): 加载NumPy数组或NPZ压缩文件
    annotations = np.load(cam_file)
    
    # 提取相机内参矩阵数组并转换为float32类型
    # .astype(np.float32): 确保数据类型一致性，节省内存
    cam_ints = annotations["intrinsics"].astype(np.float32)
    
    # 提取相机外参矩阵数组并转换为float32类型
    # 外参矩阵通常是4x4的齐次变换矩阵，表示相机在世界坐标系中的位置和方向
    cam_exts = annotations["extrinsics"].astype(np.float32)

    # =========================================================================
    # 获取图像和深度文件列表 (Get Image and Depth File Lists)
    # =========================================================================
    
    # 列出并排序RGB图像文件名
    # os.listdir(): 列出目录中的所有文件和子目录
    # [f for f in ... if f.endswith(".jpg")]: 列表推导式，只保留.jpg文件
    # sorted(): 按字母顺序排序，确保处理顺序一致
    rgbs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    
    # 列出并排序深度图文件名
    depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

    # =========================================================================
    # 验证数据一致性 (Verify Data Consistency)
    # =========================================================================
    
    # 确保内参、外参、RGB图像和深度图的数量完全一致
    # len(): 获取列表或数组的长度
    # 如果数量不匹配，说明数据集不完整或有问题
    if not (len(cam_ints) == len(cam_exts) == len(rgbs) == len(depths)):
        # 抛出值错误异常，提供详细的不匹配信息
        raise ValueError(
            f"Mismatch in sequence {seq_dir}: "
            f"{len(cam_ints)} intrinsics, {len(cam_exts)} extrinsics, "
            f"{len(rgbs)} images, {len(depths)} depths."
        )

    # =========================================================================
    # 检查是否已处理 (Check if Already Processed)
    # =========================================================================
    
    # 通过比较输出目录中的文件数量和输入图像数量来判断是否已处理
    # 如果数量相等，说明该序列已经处理过，可以跳过
    if len(os.listdir(out_img_dir)) == len(rgbs):
        return  # 提前返回，跳过已处理的序列

    # =========================================================================
    # 逐帧处理 (Frame-by-Frame Processing)
    # =========================================================================
    
    # 使用tqdm显示进度条，遍历处理每一帧
    # range(len(cam_exts)): 创建从0到帧数-1的索引序列
    # desc="Processing frames": 进度条描述文本
    # leave=False: 处理完成后不保留进度条
    for i in tqdm(range(len(cam_exts)), desc="Processing frames", leave=False):
        
        # =====================================================================
        # 验证文件名一致性 (Verify Filename Consistency)
        # =====================================================================
        
        # 从RGB图像文件名中提取帧索引
        # .split(".")[0]: 去掉文件扩展名
        # .split("_")[-1]: 取下划线分割后的最后一部分（假设格式为prefix_index.jpg）
        basename_img = rgbs[i].split(".")[0].split("_")[-1]
        
        # 从深度图文件名中提取帧索引
        basename_depth = depths[i].split(".")[0].split("_")[-1]
        
        # 验证RGB图像和深度图的帧索引是否与循环索引一致
        # int(): 将字符串转换为整数进行比较
        if int(basename_img) != i or int(basename_depth) != i:
            # 如果索引不匹配，抛出值错误异常
            raise ValueError(
                f"Frame index mismatch in sequence {seq_dir} for frame {i}"
            )

        # =====================================================================
        # 构建当前帧的文件路径 (Build Current Frame File Paths)
        # =====================================================================
        
        # 构建当前帧RGB图像的完整路径
        img_path = os.path.join(img_dir, rgbs[i])
        
        # 构建当前帧深度图的完整路径
        depth_path = os.path.join(depth_dir, depths[i])

        # =====================================================================
        # 处理相机参数 (Process Camera Parameters)
        # =====================================================================
        
        # 获取当前帧的相机内参矩阵
        # 内参矩阵包含焦距、主点等信息，用于像素坐标和相机坐标的转换
        intrins = cam_ints[i]
        
        # 获取当前帧的相机外参矩阵
        # 外参矩阵描述相机相对于世界坐标系的位置和方向
        cam_extrinsic = cam_exts[i]
        
        # 计算相机位姿（通过反转外参矩阵）
        # np.linalg.inv(): 计算矩阵的逆
        # 外参通常是世界到相机的变换，逆矩阵是相机到世界的变换（位姿）
        pose = np.linalg.inv(cam_extrinsic)
        
        # 验证计算得到的位姿矩阵是否有效
        # np.any(): 检查数组中是否有任何元素满足条件
        # np.isinf(): 检查是否为无穷大
        # np.isnan(): 检查是否为非数字（NaN）
        if np.any(np.isinf(pose)) or np.any(np.isnan(pose)):
            # 如果位姿矩阵包含无效值，抛出值错误异常
            raise ValueError(
                f"Invalid pose computed from extrinsics for frame {i} in {seq_dir}"
            )

        # =====================================================================
        # 处理深度图 (Process Depth Map)
        # =====================================================================
        
        # 使用OpenCV读取16位深度图像
        # cv2.imread(): OpenCV的图像读取函数
        # cv2.IMREAD_ANYDEPTH: 保持原始位深度，对于深度图很重要
        depth_16bit = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        
        # 将16位深度值转换为米单位的浮点数
        # .astype(np.float32): 转换数据类型为32位浮点数
        # / 65535.0: 将16位整数（0-65535）归一化到0-1范围
        # * 1000.0: 转换为毫米，然后除以1000转换为米（实际上这里是归一化后再乘1000）
        depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0

        # =====================================================================
        # 保存处理后的数据 (Save Processed Data)
        # =====================================================================
        
        # 使用RGB图像的基础文件名作为输出文件的基础名
        basename = basename_img  # 也可以使用 str(i) 作为文件名
        
        # 构建RGB图像输出路径
        out_img_path = os.path.join(out_img_dir, basename + ".jpg")
        
        # 复制RGB图像到输出目录
        # shutil.copyfile(): 复制文件内容，保持原始文件不变
        shutil.copyfile(img_path, out_img_path)
        
        # 保存深度图为NumPy数组
        # np.save(): 将数组保存为.npy格式的二进制文件
        # 这种格式加载速度快，适合深度学习应用
        np.save(os.path.join(out_depth_dir, basename + ".npy"), depth)
        
        # 保存相机参数（内参和位姿）
        # np.savez(): 将多个数组保存到一个压缩的.npz文件中
        # intrinsics=intrins: 保存内参矩阵
        # pose=pose: 保存相机位姿矩阵
        np.savez(
            os.path.join(out_cam_dir, basename + ".npz"), 
            intrinsics=intrins, 
            pose=pose
        )


def process_split(split_dir, out_split_dir):
    """
    处理数据分割中的所有序列（如train, test或val）
    
    这个函数遍历一个数据分割目录中的所有序列，并调用process_sequence
    函数处理每个序列。
    
    参数说明：
        split_dir (str): 分割目录路径
        out_split_dir (str): 处理后分割的输出目录路径
    """
    
    # =========================================================================
    # 获取序列列表 (Get Sequence List)
    # =========================================================================
    
    # 列出分割目录中的所有子目录（序列）
    # os.path.isdir(): 检查路径是否为目录
    # os.path.join(split_dir, d): 构建完整的目录路径
    # [d for d in ... if os.path.isdir(...)]: 只保留目录，过滤掉文件
    # sorted(): 按字母顺序排序，确保处理顺序一致
    sequences = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )
    
    # =========================================================================
    # 处理每个序列 (Process Each Sequence)
    # =========================================================================
    
    # 使用tqdm显示进度条，遍历所有序列
    # desc=f"Processing sequences in {os.path.basename(split_dir)}": 
    #   显示当前正在处理的分割名称
    # os.path.basename(): 获取路径的最后一部分（目录名）
    for seq in tqdm(
        sequences, desc=f"Processing sequences in {os.path.basename(split_dir)}"
    ):
        # 构建当前序列的完整输入路径
        seq_dir = os.path.join(split_dir, seq)
        
        # 构建当前序列的完整输出路径
        out_seq_dir = os.path.join(out_split_dir, seq)
        
        # 调用序列处理函数处理当前序列
        process_sequence(seq_dir, out_seq_dir)


def main():
    """
    主函数：解析命令行参数并启动数据集处理流程
    
    这个函数是程序的入口点，负责：
    1. 设置命令行参数解析
    2. 检查输入目录的有效性
    3. 为每个数据分割调用处理函数
    """
    
    # =========================================================================
    # 命令行参数设置 (Command Line Argument Setup)
    # =========================================================================
    
    # 创建参数解析器对象
    # description: 程序功能描述，显示在帮助信息中
    parser = argparse.ArgumentParser(
        description="Preprocess Point Odyssey dataset by processing images, depth maps, and camera parameters."
    )
    
    # 添加输入目录参数
    # --input_dir: 参数名称
    # type=str: 参数类型为字符串
    # required=True: 该参数为必需参数
    # help: 参数说明文本
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the root input dataset directory.",
    )
    
    # 添加输出目录参数
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the root output directory where processed data will be stored.",
    )
    
    # 解析命令行参数
    # parse_args(): 解析sys.argv中的命令行参数
    args = parser.parse_args()

    # =========================================================================
    # 数据分割处理 (Data Split Processing)
    # =========================================================================
    
    # 定义预期的数据集分割类型
    # Point Odyssey数据集通常包含训练、测试和验证三个分割
    splits = ["train", "test", "val"]
    
    # 遍历每个数据分割
    for split in splits:
        # 构建当前分割的输入目录路径
        split_dir = os.path.join(args.input_dir, split)
        
        # 构建当前分割的输出目录路径
        out_split_dir = os.path.join(args.output_dir, split)
        
        # 检查分割目录是否存在
        if not os.path.exists(split_dir):
            # 如果分割目录不存在，打印警告信息并跳过
            print(
                f"Warning: Split directory {split_dir} does not exist. Skipping this split."
            )
            continue  # 跳过当前分割，继续处理下一个
        
        # 创建输出分割目录
        # exist_ok=True: 如果目录已存在，不会报错
        os.makedirs(out_split_dir, exist_ok=True)
        
        # 调用分割处理函数
        process_split(split_dir, out_split_dir)


# =============================================================================
# 程序入口点 (Program Entry Point)
# =============================================================================

if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用main函数
    # 这是Python的标准做法，确保只有直接运行脚本时才执行主逻辑
    main()


"""
=============================================================================
                              Point Odyssey数据处理流程总结
=============================================================================

1. 输入数据结构：
   root_dir/
   ├── train/
   │   ├── sequence1/
   │   │   ├── rgbs/          # RGB图像文件夹
   │   │   │   ├── image_0.jpg
   │   │   │   ├── image_1.jpg
   │   │   │   └── ...
   │   │   ├── depths/        # 深度图文件夹
   │   │   │   ├── depth_0.png
   │   │   │   ├── depth_1.png
   │   │   │   └── ...
   │   │   └── anno.npz       # 相机参数文件
   │   └── sequence2/
   │       └── ...
   ├── test/
   └── val/

2. 输出数据结构：
   output_dir/
   ├── train/
   │   ├── sequence1/
   │   │   ├── rgb/           # 处理后的RGB图像
   │   │   │   ├── 0.jpg
   │   │   │   ├── 1.jpg
   │   │   │   └── ...
   │   │   ├── depth/         # 处理后的深度图（NPY格式）
   │   │   │   ├── 0.npy
   │   │   │   ├── 1.npy
   │   │   │   └── ...
   │   │   └── cam/           # 相机参数（NPZ格式）
   │   │       ├── 0.npz
   │   │       ├── 1.npz
   │   │       └── ...
   │   └── sequence2/
   │       └── ...
   ├── test/
   └── val/

3. 关键处理步骤：
   a) 数据验证：检查输入文件完整性和一致性
   b) 目录创建：创建标准化的输出目录结构
   c) 相机参数处理：从外参计算位姿矩阵
   d) 深度图转换：16位PNG → 32位浮点数NPY
   e) 文件复制：RGB图像直接复制
   f) 参数保存：内参和位姿保存为NPZ格式

4. 数据格式说明：
   - RGB图像：JPG格式，直接复制
   - 深度图：从16位PNG转换为32位浮点数NPY，单位为米
   - 相机内参：3x3矩阵，包含焦距和主点信息
   - 相机位姿：4x4矩阵，表示相机到世界坐标的变换

5. 错误处理：
   - 文件缺失检查
   - 数据一致性验证
   - 位姿矩阵有效性检查
   - 详细的错误信息提示

=============================================================================
                              与你的数据集对比
=============================================================================

Point Odyssey格式 vs 你的数据集：

相似点：
✅ 单目相机（每帧一个视角）
✅ 深度数据为NPY格式，单位为米
✅ 多个视频序列结构
✅ 相机内外参分开存储

不同点：
❌ 内参格式：NPZ数组 vs JSON文件
❌ 外参格式：NPZ外参矩阵 vs JSON位姿
❌ 文件结构：anno.npz vs 分离的JSON文件

适配建议：
如果选择Point Odyssey作为基础，需要修改：
1. 内参读取：从JSON而不是NPZ加载
2. 外参处理：直接使用位姿而不是反转外参
3. 文件路径：适配你的目录结构

=============================================================================
""" 