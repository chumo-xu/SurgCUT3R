#!/usr/bin/env python3
"""
CUT3R在线3D点云推理和可视化脚本 - 详细解释版本

这个脚本是CUT3R模型的在线推理演示程序，主要功能包括：
1. 从图像序列或视频文件中加载输入数据
2. 使用ARCroco3DStereo模型进行3D重建推理
3. 生成3D点云、深度图和相机位姿
4. 通过PointCloudViewer进行交互式3D可视化

CUT3R (Continuous 3D Perception Model with Persistent State) 是一个基于
Transformer的连续3D感知模型，能够从多视角图像序列中重建3D场景。

使用方法:
    python demo_online.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                         [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

示例:
    python demo_online.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512

作者注释：本文件是demo_online.py的详细解释版本，每行代码都有详细的中文注释
"""

# ==================== 导入必要的库 ====================

import argparse
# argparse: Python标准库，用于解析命令行参数
# 允许用户通过命令行传递参数来控制程序行为

import os
# os: 操作系统接口模块，提供与操作系统交互的功能
# 主要用于文件路径操作、目录创建等

import sys
# sys: 系统相关的参数和函数模块
# 这里主要用于修改Python模块搜索路径

# Add the 'src' directory to the Python path
# 将'src'目录添加到Python模块搜索路径中
# 这是为了能够导入src目录下的dust3r等模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# 解释：
# - __file__: 当前脚本文件的路径
# - os.path.dirname(__file__): 获取当前脚本所在的目录
# - os.path.join(..., 'src'): 拼接得到src目录的路径
# - os.path.abspath(): 转换为绝对路径
# - sys.path.insert(0, ...): 将路径插入到模块搜索路径的最前面

import torch
# PyTorch深度学习框架，用于：
# - 张量计算和GPU加速
# - 神经网络模型的加载和推理
# - 自动微分（虽然在推理时不需要）

import numpy as np
# NumPy数值计算库，用于：
# - 高效的数组操作
# - 数学计算和线性代数
# - 与PyTorch张量的相互转换

import cv2
# OpenCV计算机视觉库，用于：
# - 图像和视频的读取、处理
# - 视频帧的提取
# - 图像格式转换

from PIL import Image
# PIL (Python Imaging Library) 图像处理库
# 用于图像的加载、保存和基本处理操作

import time
# 时间相关的函数模块
# 用于测量推理时间和性能分析

import glob
# 文件路径模式匹配模块
# 用于根据通配符模式查找文件（如*.jpg）

import random
# 随机数生成模块
# 用于设置随机种子，确保结果的可重现性

import tempfile
# 临时文件和目录管理模块
# 用于处理视频时创建临时目录存储提取的帧

import shutil
# 高级文件操作模块
# 用于目录的复制、移动和删除操作

from copy import deepcopy
# 深拷贝函数，用于创建对象的完全独立副本
# 在处理视图数据时避免意外的引用修改

from add_ckpt_path import add_path_to_dust3r
# 自定义模块，用于添加模型检查点路径到Python搜索路径
# 这是为了正确导入dust3r相关模块所必需的

import imageio.v2 as iio
# imageio图像I/O库，用于读写各种图像格式
# v2是新版本的API，提供更好的性能和功能

# Set random seed for reproducibility.
# 设置随机种子以确保结果的可重现性
random.seed(42)
# 42是一个常用的随机种子值（来自《银河系漫游指南》中"生命、宇宙以及一切的答案"）
# 设置固定种子确保每次运行程序时随机操作的结果都相同


def parse_args():
    """
    解析命令行参数的函数
    
    这个函数定义了程序接受的所有命令行参数，包括：
    - 模型路径：预训练模型的检查点文件路径
    - 序列路径：输入图像序列或视频文件的路径
    - 设备选择：使用CPU还是GPU进行计算
    - 图像尺寸：输入图像的目标尺寸
    - 可视化阈值：点云可视化的置信度阈值
    - 输出目录：保存结果的目录路径
    
    返回:
        argparse.Namespace: 包含所有解析后参数的对象
    """
    # 创建参数解析器对象
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
        # description: 程序的简短描述，会在帮助信息中显示
    )
    
    # 添加模型路径参数
    parser.add_argument(
        "--model_path",                              # 参数名称（长格式）
        type=str,                                   # 参数类型为字符串
        default="src/cut3r_512_dpt_4_64.pth",      # 默认值：512分辨率的DPT头模型
        help="Path to the pretrained model checkpoint.",  # 帮助信息
    )
    # 解释：CUT3R提供两种预训练模型：
    # - cut3r_224_linear_4.pth: 224分辨率，线性头，适合快速推理
    # - cut3r_512_dpt_4_64.pth: 512分辨率，DPT头，质量更高
    
    # 添加序列路径参数
    parser.add_argument(
        "--seq_path",                               # 参数名称
        type=str,                                   # 参数类型
        default="",                                 # 默认为空，需要用户指定
        help="Path to the directory containing the image sequence.",  # 帮助信息
    )
    # 解释：可以是包含图像文件的目录，也可以是视频文件路径
    
    # 添加设备参数
    parser.add_argument(
        "--device",                                 # 参数名称
        type=str,                                   # 参数类型
        default="cuda",                             # 默认使用GPU
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",  # 帮助信息
    )
    # 解释：
    # - "cuda": 使用GPU加速（需要NVIDIA GPU和CUDA支持）
    # - "cpu": 使用CPU计算（较慢但兼容性好）
    
    # 添加图像尺寸参数
    parser.add_argument(
        "--size",                                   # 参数名称
        type=int,                                   # 参数类型为整数
        default="512",                              # 默认512像素
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    # 解释：输入图像会被缩放到这个尺寸的正方形
    # 必须与所选模型匹配：224模型用224，512模型用512
    
    # 添加可视化阈值参数
    parser.add_argument(
        "--vis_threshold",                          # 参数名称
        type=float,                                 # 参数类型为浮点数
        default=1.5,                                # 默认阈值1.5
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    # 解释：控制点云可视化的置信度阈值
    # 只有置信度高于此阈值的3D点才会被显示
    # 较高的阈值会显示更少但更可靠的点
    
    # 添加输出目录参数
    parser.add_argument(
        "--output_dir",                             # 参数名称
        type=str,                                   # 参数类型
        default="./demo_tmp",                       # 默认输出目录
        help="value for tempfile.tempdir",         # 帮助信息
    )
    # 解释：推理结果（深度图、置信度图、相机参数等）的保存目录
    
    # 解析并返回所有参数
    return parser.parse_args()
    # parse_args()会自动解析sys.argv中的命令行参数
    # 返回一个Namespace对象，可以通过属性访问各个参数值


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    为推理准备输入视图数据的核心函数

    这个函数是CUT3R推理流程中的关键步骤，负责将原始图像路径转换为
    模型可以处理的标准化视图数据结构。它支持两种输入模式：
    1. 纯图像模式：只使用RGB图像进行推理
    2. 混合模式：同时使用图像和射线图（ray maps）

    CUT3R模型的核心思想是通过多视角几何约束来重建3D场景，因此
    每个视图都需要包含完整的几何和外观信息。

    参数详解:
        img_paths (list): 图像文件路径列表，每个元素是一个图像文件的完整路径
        img_mask (list of bool): 图像有效性掩码，True表示对应位置有有效图像
        size (int): 目标图像尺寸，所有图像会被缩放到size×size的正方形
        raymaps (list, optional): 射线图列表，包含从参考视图到当前视图的几何变换信息
        raymap_mask (list, optional): 射线图有效性掩码，True表示对应位置有有效射线图
        revisit (int): 重访次数，控制每个视图被处理多少次（用于数据增强）
        update (bool): 是否在重访时更新模型状态（用于序列处理）

    返回:
        list: 视图字典列表，每个字典包含模型推理所需的所有信息
    """

    # Import image loader (delayed import needed after adding ckpt path).
    # 延迟导入图像加载器（需要在添加检查点路径后才能导入）
    from src.dust3r.utils.image import load_images
    # 解释：这个导入必须放在函数内部，因为dust3r模块的路径是在运行时
    # 通过add_path_to_dust3r()函数动态添加的，如果在文件开头导入会失败

    # 使用dust3r的图像加载器加载所有图像
    images = load_images(img_paths, size=size)
    # load_images函数的功能：
    # 1. 读取每个路径对应的图像文件
    # 2. 将图像缩放到指定尺寸（size×size）
    # 3. 进行标准化处理（转换为[-1,1]范围的浮点数）
    # 4. 转换为PyTorch张量格式
    # 5. 记录原始图像的真实尺寸信息

    # 初始化视图列表
    views = []
    # views将存储所有处理后的视图数据，每个元素是一个包含完整视图信息的字典

    # 检查是否只提供了图像（没有射线图）
    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        # 纯图像模式：只使用RGB图像进行推理

        # 遍历所有加载的图像，为每个图像创建视图数据结构
        for i in range(len(images)):
            # 为第i个图像创建视图字典
            view = {
                # === 核心数据 ===
                "img": images[i]["img"],
                # 图像张量，形状为(batch_size, 3, height, width)
                # 像素值范围为[-1, 1]，已经过标准化处理

                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],    # batch_size维度
                        6,                            # 射线图通道数：3(起点) + 3(方向)
                        images[i]["img"].shape[-2],   # 高度维度
                        images[i]["img"].shape[-1],   # 宽度维度
                    ),
                    torch.nan,                        # 填充NaN值表示无效
                ),
                # 射线图张量：在纯图像模式下，射线图被设置为NaN
                # 射线图包含从参考视图到当前视图的几何变换信息
                # 6个通道分别表示：射线起点(x,y,z) + 射线方向(dx,dy,dz)

                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                # 图像的真实尺寸，形状为(2,)，包含[height, width]
                # 这个信息用于后续的几何计算和坐标变换

                # === 索引和标识信息 ===
                "idx": i,
                # 视图在序列中的索引，用于追踪和调试

                "instance": str(i),
                # 实例标识符，通常用于区分不同的视图实例

                # === 相机参数 ===
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                # 相机位姿矩阵，4×4的齐次变换矩阵
                # 在纯图像模式下初始化为单位矩阵（表示世界坐标系）
                # unsqueeze(0)添加batch维度，最终形状为(1, 4, 4)

                # === 控制掩码 ===
                "img_mask": torch.tensor(True).unsqueeze(0),
                # 图像掩码：True表示使用图像数据进行推理
                # unsqueeze(0)添加batch维度

                "ray_mask": torch.tensor(False).unsqueeze(0),
                # 射线图掩码：False表示不使用射线图数据
                # 在纯图像模式下总是False

                # === 序列控制参数 ===
                "update": torch.tensor(True).unsqueeze(0),
                # 更新标志：True表示这个视图会更新模型的内部状态
                # 用于CUT3R的连续状态管理机制

                "reset": torch.tensor(False).unsqueeze(0),
                # 重置标志：False表示不重置模型状态，保持序列连续性
                # True通常只在序列开始时使用
            }
            # 将构建好的视图字典添加到视图列表中
            views.append(view)

    else:
        # Combine images and raymaps.
        # 混合模式：同时使用图像和射线图进行推理
        # 这种模式通常用于更精确的几何重建，特别是在已知相机参数的情况下

        # 计算总视图数量
        num_views = len(images) + len(raymaps)
        # 总视图数 = 图像数量 + 射线图数量

        # 验证掩码长度的一致性
        assert len(img_mask) == len(raymap_mask) == num_views
        # 确保图像掩码和射线图掩码的长度都等于总视图数

        # 验证掩码中True值的数量与实际数据数量匹配
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)
        # sum(img_mask): 计算图像掩码中True的数量，应该等于图像数量
        # sum(raymap_mask): 计算射线图掩码中True的数量，应该等于射线图数量

        # 初始化索引计数器
        j = 0  # 图像索引计数器
        k = 0  # 射线图索引计数器

        # 遍历所有视图位置，根据掩码决定使用图像还是射线图
        for i in range(num_views):
            # 为第i个视图位置创建视图字典
            view = {
                # === 图像数据处理 ===
                "img": (
                    images[j]["img"]                    # 如果img_mask[i]为True，使用第j个图像
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)  # 否则创建NaN填充的张量
                ),
                # 解释：根据掩码决定是使用真实图像还是NaN占位符
                # torch.full_like()创建与images[0]["img"]相同形状但填充NaN的张量

                # === 射线图数据处理 ===
                "ray_map": (
                    raymaps[k]                          # 如果raymap_mask[i]为True，使用第k个射线图
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)  # 否则创建NaN填充的张量
                ),
                # 解释：根据掩码决定是使用真实射线图还是NaN占位符

                # === 真实尺寸信息处理 ===
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])  # 如果使用图像，获取图像的真实尺寸
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))  # 否则从射线图推导尺寸
                ),
                # 解释：
                # - 如果使用图像：直接使用图像的true_shape信息
                # - 如果使用射线图：从射线图的形状推导尺寸
                #   raymaps[k].shape[1:-1] 获取中间维度（去掉batch和channel维度）
                #   [::-1] 反转顺序（从[H,W]变为[W,H]，然后转为[H,W]）

                # === 其他标准字段 ===
                "idx": i,                               # 视图索引
                "instance": str(i),                     # 实例标识
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),  # 相机位姿

                # === 掩码信息 ===
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),      # 图像掩码
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),   # 射线图掩码

                # === 序列控制 ===
                "update": torch.tensor(img_mask[i]).unsqueeze(0),        # 更新标志（通常与图像掩码相同）
                "reset": torch.tensor(False).unsqueeze(0),               # 重置标志
            }

            # 更新索引计数器
            if img_mask[i]:
                j += 1  # 如果使用了图像，图像索引递增
            if raymap_mask[i]:
                k += 1  # 如果使用了射线图，射线图索引递增

            # 将视图添加到列表中
            views.append(view)

        # 验证所有图像和射线图都被使用了
        assert j == len(images) and k == len(raymaps)
        # 确保图像索引j等于图像总数，射线图索引k等于射线图总数
        # 这个断言确保没有遗漏任何数据

    # === 重访逻辑处理 ===
    # 重访（revisit）是CUT3R中的一个重要概念，用于数据增强和状态更新
    if revisit > 1:
        # 如果重访次数大于1，需要复制视图数据
        new_views = []  # 创建新的视图列表

        # 外层循环：重访次数
        for r in range(revisit):
            # 内层循环：原始视图
            for i, view in enumerate(views):
                # 创建视图的深拷贝，避免修改原始数据
                new_view = deepcopy(view)
                # deepcopy确保所有嵌套对象都被完全复制

                # 更新索引信息
                new_view["idx"] = r * len(views) + i
                # 新索引 = 重访轮次 × 原始视图数 + 原始索引
                # 例如：第2轮重访的第3个视图，索引为 1*N + 2

                new_view["instance"] = str(r * len(views) + i)
                # 更新实例标识符，保持与索引一致

                # 处理更新标志
                if r > 0 and not update:
                    # 如果是重访（r > 0）且不允许更新，则设置update为False
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                    # 这意味着重访的视图不会更新模型状态，只用于推理

                # 将处理后的视图添加到新列表中
                new_views.append(new_view)

        # 返回包含重访数据的新视图列表
        return new_views

    # 如果不需要重访，直接返回原始视图列表
    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    处理模型推理输出，生成用于可视化的3D点云和相机参数

    这个函数是CUT3R推理流程的后处理阶段，负责将模型的原始输出转换为
    可视化系统能够理解的格式。主要包括：
    1. 提取和处理3D点云数据
    2. 恢复相机位姿信息
    3. 估计相机内参（焦距等）
    4. 保存中间结果到文件
    5. 准备可视化所需的数据结构

    CUT3R模型输出包含多种信息：
    - pts3d_in_self_view: 在自身视角坐标系下的3D点
    - pts3d_in_other_view: 在其他视角坐标系下的3D点
    - camera_pose: 相机位姿编码
    - conf_self/conf: 置信度信息
    - img: 重建的RGB图像

    参数详解:
        outputs (dict): 模型推理的原始输出，包含pred和views两个主要部分
        outdir (str): 输出目录路径，用于保存中间结果文件
        revisit (int): 重访次数，用于确定有效输出的长度
        use_pose (bool): 是否使用相机位姿进行坐标变换

    返回:
        tuple: (3D点云列表, 颜色列表, 置信度列表, 相机参数字典)
    """

    # 导入必要的工具函数
    from src.dust3r.utils.camera import pose_encoding_to_camera
    # pose_encoding_to_camera: 将模型输出的位姿编码转换为标准的4×4变换矩阵

    from src.dust3r.post_process import estimate_focal_knowing_depth
    # estimate_focal_knowing_depth: 基于深度信息估计相机焦距的函数
    # 这是一个重要的后处理步骤，因为CUT3R可能没有准确的相机内参

    from src.dust3r.utils.geometry import geotrf
    # geotrf: 几何变换函数，用于将3D点从一个坐标系变换到另一个坐标系

    # === 第一步：处理重访数据，只保留最后一轮完整的输出 ===
    # Only keep the outputs corresponding to one full pass.
    # 只保留对应一次完整遍历的输出
    valid_length = len(outputs["pred"]) // revisit
    # 计算有效长度：总输出数量除以重访次数
    # 例如：如果有12个输出，重访3次，则有效长度为4

    outputs["pred"] = outputs["pred"][-valid_length:]
    # 只保留最后valid_length个预测结果
    # 使用负索引[-valid_length:]获取列表的最后部分

    outputs["views"] = outputs["views"][-valid_length:]
    # 同样只保留最后valid_length个视图数据
    # 确保pred和views的数量保持一致

    # === 第二步：提取3D点云和置信度信息 ===
    # 提取自视角3D点云（每个视图在自己坐标系下的3D点）
    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    # 遍历所有预测输出，提取pts3d_in_self_view字段
    # .cpu()将张量从GPU移动到CPU，便于后续处理
    # 形状：每个元素为(batch_size, height, width, 3)

    # 提取跨视角3D点云（在其他视图坐标系下的3D点）
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    # pts3d_in_other_view包含了几何一致性约束下的3D点
    # 这些点考虑了多视角之间的几何关系

    # 提取自视角置信度
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    # conf_self表示模型对自视角3D重建结果的置信度
    # 形状：每个元素为(batch_size, height, width)

    # 提取跨视角置信度
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    # conf表示模型对跨视角几何一致性的置信度
    # 用于评估多视角重建的可靠性

    # 将自视角点云拼接成一个大张量
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)
    # torch.cat在第0维（batch维）上拼接所有点云
    # 最终形状：(total_batch_size, height, width, 3)

    # === 第三步：恢复相机位姿信息 ===
    # Recover camera poses.
    # 恢复相机位姿
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]
    # 遍历所有预测结果，提取并转换相机位姿：
    # 1. pred["camera_pose"]: 模型输出的位姿编码（通常是7维：3位置+4四元数）
    # 2. .clone(): 创建张量副本，避免修改原始数据
    # 3. pose_encoding_to_camera(): 将编码转换为4×4变换矩阵
    # 4. .cpu(): 移动到CPU进行后续处理

    # 提取旋转矩阵（相机到世界坐标系）
    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    # 从每个4×4位姿矩阵中提取左上角3×3旋转矩阵
    # torch.cat在batch维上拼接所有旋转矩阵
    # 最终形状：(total_views, 3, 3)

    # 提取平移向量（相机到世界坐标系）
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)
    # 从每个4×4位姿矩阵中提取右侧3×1平移向量
    # 最终形状：(total_views, 3)

    # === 第四步：处理坐标变换（可选） ===
    if use_pose:
        # 如果启用位姿变换，将自视角点云变换到世界坐标系
        transformed_pts3ds_other = []

        # 遍历每个位姿和对应的自视角点云
        for pose, pself in zip(pr_poses, pts3ds_self):
            # 使用几何变换函数将点云从相机坐标系变换到世界坐标系
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
            # geotrf(pose, points): 应用位姿变换
            # pself.unsqueeze(0): 添加batch维度以匹配geotrf的输入要求

        # 使用变换后的点云替换原始的跨视角点云
        pts3ds_other = transformed_pts3ds_other
        # 同时使用自视角的置信度（因为变换后的点云基于自视角）
        conf_other = conf_self

    # === 第五步：估计相机内参 ===
    # Estimate focal length based on depth.
    # 基于深度信息估计焦距
    B, H, W, _ = pts3ds_self.shape
    # 获取点云张量的维度：B=batch_size, H=height, W=width, _=3(xyz坐标)

    # 设置主点（图像中心）
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    # 主点通常假设在图像中心：(width/2, height/2)
    # .repeat(B, 1): 为每个batch复制相同的主点坐标
    # 最终形状：(B, 2)

    # 使用Weiszfeld算法估计焦距
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")
    # estimate_focal_knowing_depth函数：
    # - 输入：3D点云和主点坐标
    # - 输出：估计的焦距值
    # - weiszfeld模式：使用Weiszfeld算法进行鲁棒估计
    # 这个算法能够处理噪声和异常值，得到更可靠的焦距估计

    # === 第六步：提取颜色信息 ===
    # 从视图数据中提取RGB颜色信息
    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]
    # 处理步骤：
    # 1. output["img"]: 获取图像张量，形状为(B, 3, H, W)
    # 2. .permute(0, 2, 3, 1): 重排维度为(B, H, W, 3)，符合图像显示格式
    # 3. + 1.0: 将像素值从[-1,1]范围平移到[0,2]
    # 4. 0.5 * (...): 缩放到[0,1]范围，这是标准的图像像素值范围

    # === 第七步：构建相机参数字典 ===
    cam_dict = {
        "focal": focal.cpu().numpy(),           # 焦距参数
        "pp": pp.cpu().numpy(),                 # 主点坐标
        "R": R_c2w.cpu().numpy(),              # 旋转矩阵
        "t": t_c2w.cpu().numpy(),              # 平移向量
    }
    # 这个字典包含了可视化系统需要的所有相机参数
    # 所有张量都转换为numpy数组，便于后续处理

    # === 第八步：准备保存数据 ===
    # 准备要保存的各种数据，转换为合适的格式

    # 自视角3D点云（用于保存）
    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    # 提取深度信息（Z坐标）
    depths_tosave = pts3ds_self_tosave[..., 2]  # B, H, W
    # [..., 2]表示取最后一个维度的第2个元素（Z坐标）

    # 跨视角3D点云（拼接所有视图）
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3

    # 置信度信息（拼接所有视图）
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W

    # 颜色信息（重新处理并拼接）
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    # 与前面的colors处理相同，但确保在CPU上并拼接所有视图

    # 相机到世界坐标系的变换矩阵
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4

    # 构建相机内参矩阵
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    # torch.eye(3): 创建3×3单位矩阵
    # .unsqueeze(0): 添加batch维度，形状变为(1, 3, 3)
    # .repeat(...): 为每个视图复制一份，最终形状为(B, 3, 3)

    # 设置内参矩阵的具体值
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()  # fx = 焦距
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()  # fy = 焦距（假设fx=fy）
    intrinsics_tosave[:, 0, 2] = pp[:, 0]              # cx = 主点x坐标
    intrinsics_tosave[:, 1, 2] = pp[:, 1]              # cy = 主点y坐标
    # 标准的相机内参矩阵格式：
    # [[fx,  0, cx],
    #  [ 0, fy, cy],
    #  [ 0,  0,  1]]

    # === 第九步：保存中间结果到文件 ===
    # 创建输出目录结构
    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # 创建深度图保存目录，exist_ok=True表示目录已存在时不报错

    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # 创建置信度图保存目录

    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # 创建彩色图像保存目录

    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    # 创建相机参数保存目录

    # 遍历所有帧，保存各种数据
    for f_id in range(len(pts3ds_self)):
        # 提取当前帧的各种数据
        depth = depths_tosave[f_id].cpu().numpy()        # 深度图
        conf = conf_self_tosave[f_id].cpu().numpy()      # 置信度图
        color = colors_tosave[f_id].cpu().numpy()        # 彩色图像
        c2w = cam2world_tosave[f_id].cpu().numpy()       # 相机位姿
        intrins = intrinsics_tosave[f_id].cpu().numpy()  # 相机内参

        # 保存深度图（.npy格式，保持精度）
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        # f"{f_id:06d}": 格式化文件名，6位数字，不足位数用0填充
        # 例如：000000.npy, 000001.npy, ...

        # 保存置信度图（.npy格式）
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)

        # 保存彩色图像（.png格式，便于查看）
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),  # 转换为0-255范围的整数
        )
        # imageio.imwrite保存图像文件
        # color原本在[0,1]范围，乘以255转换为[0,255]范围
        # .astype(np.uint8)转换为8位无符号整数类型

        # 保存相机参数（.npz格式，可以保存多个数组）
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,        # 相机位姿矩阵
            intrinsics=intrins,  # 相机内参矩阵
        )
        # .npz是numpy的压缩格式，可以在一个文件中保存多个命名数组

    # === 第十步：返回可视化所需的数据 ===
    return pts3ds_other, colors, conf_other, cam_dict
    # 返回值说明：
    # - pts3ds_other: 3D点云列表，用于3D可视化
    # - colors: 颜色信息列表，对应每个点云的颜色
    # - conf_other: 置信度列表，用于过滤低质量的点
    # - cam_dict: 相机参数字典，用于设置可视化相机


def parse_seq_path(p):
    """
    解析序列路径，支持图像目录和视频文件两种输入格式

    这个函数是输入处理的核心，能够智能地处理两种不同的输入：
    1. 图像目录：包含多个图像文件的文件夹
    2. 视频文件：单个视频文件，需要提取帧

    对于视频文件，函数会：
    - 使用OpenCV读取视频
    - 提取所有帧并保存为临时图像文件
    - 返回图像路径列表和临时目录信息

    参数:
        p (str): 输入路径，可以是目录路径或视频文件路径

    返回:
        tuple: (图像路径列表, 临时目录名称或None)
    """

    # 检查输入路径是否为目录
    if os.path.isdir(p):
        # 如果是目录，直接列出所有文件并排序
        img_paths = sorted(glob.glob(f"{p}/*"))
        # glob.glob(f"{p}/*"): 匹配目录下的所有文件
        # sorted(): 按文件名排序，确保图像序列的正确顺序
        # 这对于视频序列很重要，因为帧的顺序影响3D重建质量

        tmpdirname = None
        # 不需要临时目录，因为图像已经存在

    else:
        # 如果不是目录，假设是视频文件，尝试用OpenCV打开
        cap = cv2.VideoCapture(p)
        # cv2.VideoCapture: OpenCV的视频捕获对象
        # 可以读取各种格式的视频文件（mp4, avi, mov等）

        # 检查视频是否成功打开
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
            # 如果无法打开，抛出异常并提供错误信息

        # 获取视频的基本信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # CAP_PROP_FPS: 视频的帧率（每秒帧数）

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # CAP_PROP_FRAME_COUNT: 视频的总帧数

        # 验证视频信息的有效性
        if video_fps == 0:
            cap.release()  # 释放视频捕获对象
            raise ValueError(f"Error: Video FPS is 0 for {p}")
            # FPS为0通常表示视频文件损坏或格式不支持

        # 设置帧提取参数
        frame_interval = 1
        # 帧间隔：1表示提取每一帧，2表示每隔一帧提取一次
        # 可以根据需要调整以控制处理的帧数

        # 生成要提取的帧索引列表
        frame_indices = list(range(0, total_frames, frame_interval))
        # range(0, total_frames, frame_interval): 从0开始，每隔frame_interval取一帧

        # 打印视频处理信息
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        # 向用户显示视频的基本信息和处理计划

        # 初始化图像路径列表
        img_paths = []

        # 创建临时目录存储提取的帧
        tmpdirname = tempfile.mkdtemp()
        # tempfile.mkdtemp(): 创建一个临时目录
        # 返回目录的完整路径，程序结束后需要手动清理

        # 遍历所有要提取的帧索引
        for i in frame_indices:
            # 设置视频读取位置到第i帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            # CAP_PROP_POS_FRAMES: 设置当前帧位置

            # 读取当前帧
            ret, frame = cap.read()
            # ret: 布尔值，表示是否成功读取
            # frame: 读取到的帧数据（numpy数组）

            # 检查是否成功读取帧
            if not ret:
                break  # 如果读取失败，停止处理
                # 这可能发生在视频文件末尾或读取错误时

            # 构建帧文件的保存路径
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            # 文件名格式：frame_0.jpg, frame_1.jpg, ...

            # 保存帧为JPEG图像文件
            cv2.imwrite(frame_path, frame)
            # cv2.imwrite: 保存图像到文件
            # frame: BGR格式的图像数据

            # 将帧路径添加到列表中
            img_paths.append(frame_path)

        # 释放视频捕获对象，释放系统资源
        cap.release()

    # 返回图像路径列表和临时目录信息
    return img_paths, tmpdirname
    # img_paths: 所有图像文件的路径列表
    # tmpdirname: 临时目录名称（如果创建了）或None


def run_inference(args):
    """
    执行完整的推理和可视化流程

    这是整个程序的核心函数，协调所有组件完成从输入到可视化的完整流程：
    1. 设备配置和环境准备
    2. 模型加载和初始化
    3. 输入数据准备和预处理
    4. 模型推理执行
    5. 输出后处理和保存
    6. 3D可视化启动

    CUT3R的推理流程特点：
    - 支持递归推理（recurrent inference），维护跨帧状态
    - 能够处理任意长度的图像序列
    - 生成高质量的3D点云和相机位姿
    - 提供实时的3D可视化界面

    参数:
        args: 解析后的命令行参数对象，包含所有配置信息
    """

    # === 第一步：设备配置和验证 ===
    # Set up the computation device.
    # 设置计算设备
    device = args.device
    # 从命令行参数获取设备设置（"cuda"或"cpu"）

    # 验证CUDA可用性
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"
        # 如果用户指定使用GPU但CUDA不可用，自动切换到CPU
        # torch.cuda.is_available(): 检查CUDA是否正确安装和配置

    # === 第二步：模型路径配置 ===
    # Add the checkpoint path (required for model imports in the dust3r package).
    # 添加检查点路径（dust3r包的模型导入所必需）
    add_path_to_dust3r(args.model_path)
    # 这个函数将模型文件所在目录添加到Python搜索路径
    # 这是因为dust3r模块需要能够找到相关的依赖文件

    # === 第三步：导入模型和推理函数 ===
    # Import model and inference functions after adding the ckpt path.
    # 在添加检查点路径后导入模型和推理函数
    from src.dust3r.inference import inference, inference_recurrent
    # inference: 标准推理函数，一次性处理所有视图
    # inference_recurrent: 递归推理函数，逐帧处理，维护状态

    from src.dust3r.model import ARCroco3DStereo
    # ARCroco3DStereo: CUT3R的主要模型类
    # 基于Transformer架构，支持多视角3D重建

    from viser_utils import PointCloudViewer
    # PointCloudViewer: 3D点云可视化工具
    # 提供交互式的3D场景浏览界面

    # === 第四步：输入数据准备 ===
    # Prepare image file paths.
    # 准备图像文件路径
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    # parse_seq_path: 解析输入路径，支持目录和视频文件
    # 返回图像路径列表和可能的临时目录名称

    # 验证是否找到了图像
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
        # 如果没有找到任何图像，打印错误信息并退出

    # 打印找到的图像数量
    print(f"Found {len(img_paths)} images in {args.seq_path}.")

    # 创建图像掩码（所有图像都标记为有效）
    img_mask = [True] * len(img_paths)
    # 在这个简单的演示中，所有图像都被认为是有效的
    # 在更复杂的应用中，可能需要根据质量评估设置掩码

    # === 第五步：视图数据准备 ===
    # Prepare input views.
    # 准备输入视图
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,    # 图像路径列表
        img_mask=img_mask,      # 图像有效性掩码
        size=args.size,         # 目标图像尺寸
        revisit=1,              # 重访次数（1表示不重访）
        update=True,            # 允许状态更新
    )
    # prepare_input函数将原始图像路径转换为模型可处理的视图数据结构

    # 清理临时目录（如果创建了）
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)
        # shutil.rmtree: 递归删除目录及其所有内容
        # 清理视频处理时创建的临时帧文件

    # === 第六步：模型加载和配置 ===
    # Load and prepare the model.
    # 加载和准备模型
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    # ARCroco3DStereo.from_pretrained: 从预训练检查点加载模型
    # .to(device): 将模型移动到指定设备（GPU或CPU）

    model.eval()
    # 设置模型为评估模式
    # 这会禁用dropout、batch normalization的训练行为
    # 确保推理结果的一致性和确定性

    # === 第七步：执行推理 ===
    # Run inference.
    # 运行推理
    print("Running inference...")
    start_time = time.time()
    # 记录推理开始时间，用于性能分析

    # 执行递归推理
    outputs, state_args = inference_recurrent(views, model, device)
    # inference_recurrent: 递归推理函数
    # - views: 准备好的视图数据列表
    # - model: 加载的CUT3R模型
    # - device: 计算设备
    # 返回值：
    # - outputs: 推理结果，包含3D点云、位姿、置信度等
    # - state_args: 模型的内部状态信息，用于可视化

    # 计算推理时间
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )
    # 显示总推理时间和平均每帧时间
    # 这些信息有助于评估模型的性能和效率

    # === 第八步：输出后处理 ===
    # Process outputs for visualization.
    # 处理输出用于可视化
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True
    )
    # prepare_output函数处理模型的原始输出：
    # - outputs: 模型推理结果
    # - args.output_dir: 输出目录，保存中间结果
    # - 1: 重访次数（与前面的revisit=1对应）
    # - True: 使用位姿变换
    # 返回值：
    # - pts3ds_other: 处理后的3D点云列表
    # - colors: 对应的颜色信息
    # - conf: 置信度信息
    # - cam_dict: 相机参数字典

    # === 第九步：数据格式转换 ===
    # Convert tensors to numpy arrays for visualization.
    # 将张量转换为numpy数组用于可视化
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    # 将每个点云张量从GPU移动到CPU并转换为numpy数组
    # 可视化系统通常使用numpy数组而不是PyTorch张量

    colors_to_vis = [c.cpu().numpy() for c in colors]
    # 同样转换颜色信息

    edge_colors = [None] * len(pts3ds_to_vis)
    # 创建边缘颜色列表，初始化为None
    # 这表示不使用特殊的边缘着色，使用默认颜色方案

    # === 第十步：启动3D可视化 ===
    # Create and run the point cloud viewer.
    # 创建并运行点云查看器
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,                      # CUT3R模型对象
        state_args,                 # 模型状态参数
        pts3ds_to_vis,             # 3D点云数据列表
        colors_to_vis,             # 颜色数据列表
        conf,                      # 置信度信息
        cam_dict,                  # 相机参数字典
        device=device,             # 计算设备
        edge_color_list=edge_colors,  # 边缘颜色列表
        show_camera=True,          # 显示相机位置
        vis_threshold=args.vis_threshold,  # 可视化阈值
        size=args.size             # 图像尺寸
    )
    # PointCloudViewer是一个交互式3D可视化工具，提供：
    # - 3D点云的实时渲染
    # - 相机轨迹的显示
    # - 交互式视角控制
    # - 置信度过滤
    # - 多种显示模式

    # 启动可视化界面
    viewer.run()
    # run()方法启动可视化界面的主循环
    # 用户可以通过鼠标和键盘与3D场景交互
    # 程序会一直运行直到用户关闭界面


def main():
    """
    主函数：程序的入口点

    这个函数负责：
    1. 解析命令行参数
    2. 验证输入参数的有效性
    3. 调用推理流程或显示帮助信息

    程序的整体流程：
    命令行参数解析 → 输入验证 → 推理执行 → 结果可视化
    """

    # 解析命令行参数
    args = parse_args()
    # parse_args()函数解析用户提供的所有命令行参数
    # 返回包含所有参数值的Namespace对象

    # 检查是否提供了序列路径
    if not args.seq_path:
        # 如果用户没有提供输入路径
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
        # 显示提示信息并退出程序
        # 建议用户使用gradio演示界面进行交互式输入
    else:
        # 如果提供了有效的输入路径，执行推理流程
        run_inference(args)
        # run_inference函数执行完整的推理和可视化流程


# === 程序入口点 ===
if __name__ == "__main__":
    """
    Python脚本的标准入口点检查

    当脚本被直接执行时（而不是被导入为模块），__name__变量的值为"__main__"
    这个检查确保只有在直接运行脚本时才会执行main()函数

    这种模式的好处：
    1. 允许脚本既可以直接运行，也可以作为模块导入
    2. 避免在导入时意外执行主程序逻辑
    3. 符合Python的最佳实践
    """
    main()
    # 调用主函数，开始程序执行

# ==================== 程序总结 ====================
"""
CUT3R在线推理演示程序总结

这个程序展示了如何使用CUT3R模型进行3D场景重建和可视化：

1. **输入处理**：
   - 支持图像目录和视频文件两种输入格式
   - 自动提取视频帧并进行预处理
   - 标准化图像尺寸和格式

2. **模型推理**：
   - 使用ARCroco3DStereo模型进行多视角3D重建
   - 支持递归推理，维护跨帧状态
   - 生成3D点云、深度图、相机位姿等多种输出

3. **后处理**：
   - 坐标变换和几何校正
   - 相机参数估计和优化
   - 置信度评估和过滤

4. **可视化**：
   - 交互式3D点云显示
   - 相机轨迹可视化
   - 实时参数调整

5. **技术特点**：
   - GPU加速支持
   - 内存高效的流式处理
   - 鲁棒的错误处理
   - 详细的性能监控

这个程序是CUT3R技术的完整演示，展现了从2D图像到3D场景重建的完整流程。
"""
