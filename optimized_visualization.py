#!/usr/bin/env python3
"""
优化版多帧可视化 - 通过采样减少点数

这个脚本的主要目的是可视化SCARED手术数据集的多帧3D重建结果。
由于原始数据点云密度很高，直接可视化会导致性能问题，因此采用采样策略减少点数。

主要功能：
1. 加载SCARED数据集的RGB图像、深度图和相机参数
2. 将深度图转换为3D点云（从相机坐标系转换到世界坐标系）
3. 通过均匀采样减少点云密度以提高可视化性能
4. 使用viser工具进行交互式3D可视化
5. 支持相机轨迹的缩放和位姿调整
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
import glob
import argparse

# 添加CUT3R源码路径到Python搜索路径
# 这样可以导入CUT3R项目中的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 导入CUT3R的可视化工具
# PointCloudViewer是CUT3R提供的交互式3D点云可视化工具
from viser_utils import PointCloudViewer

def sample_pointcloud_uniform(pts3d, colors, conf, step=5):
    """
    使用均匀采样减少点数，保持空间分布

    这个函数的作用是减少点云的密度以提高可视化性能。
    采用均匀采样策略，即在图像的每个step×step的网格中只取一个点。
    这样可以保持点云的整体空间分布，同时大幅减少点的数量。

    参数:
        pts3d: (H, W, 3) 3D点云数组，每个像素对应一个3D点坐标
        colors: (H, W, 3) 颜色数组，每个像素对应RGB颜色值
        conf: (H, W) 置信度数组，表示每个点的可靠程度
        step: 采样步长，每隔step个像素取一个点

    返回:
        sampled_pts: 采样后的3D点云
        sampled_colors: 采样后的颜色
        sampled_conf: 采样后的置信度
    """
    H, W = pts3d.shape[:2]  # 获取原始图像的高度和宽度

    # 均匀采样 - 使用Python的切片语法[::step]每隔step个像素取一个
    # 这相当于在H×W的网格上每隔step行和step列取一个点
    sampled_pts = pts3d[::step, ::step, :]      # 采样3D点
    sampled_colors = colors[::step, ::step, :]  # 采样对应的颜色
    sampled_conf = conf[::step, ::step]         # 采样对应的置信度

    # 计算采样后的尺寸和压缩比例
    new_h, new_w = sampled_pts.shape[:2]
    reduction_ratio = (new_h * new_w) / (H * W)  # 计算点数压缩比例

    print(f"     均匀采样: {H}x{W} -> {new_h}x{new_w} (比例: {reduction_ratio:.2%})")

    return sampled_pts, sampled_colors, sampled_conf

def load_optimized_data(data_dir, num_frames=20, sample_ratio=0.05, external_depth_npz=None, depth_scale=1.0, pose_format='c2w'):
    """
    加载优化的多帧数据

    这个函数是整个可视化流程的核心，负责：
    1. 从SCARED数据集中加载多帧的RGB图像、深度图和相机参数
    2. 将深度图转换为3D点云
    3. 进行坐标系变换（从相机坐标系到世界坐标系）
    4. 对点云进行采样以减少数据量

    参数:
        data_dir: SCARED数据集目录路径
        num_frames: 要加载的帧数
        sample_ratio: 每帧的采样比例 (0.05 = 5%，即只保留5%的点)
        external_depth_npz: 外部深度npz文件路径（可选）
        depth_scale: 深度和位姿缩放因子（可选，默认1.0不缩放）
        pose_format: 输入位姿格式，'c2w'表示camera-to-world，'w2c'表示world-to-camera（默认'c2w'）

    返回:
        pts3ds_list: 每帧的3D点云列表
        colors_list: 每帧的颜色列表
        conf_list: 每帧的置信度列表
        poses_list: 每帧的相机位姿列表
        intrinsics_list: 每帧的相机内参列表
    """
    print(f"🔍 加载优化数据 (采样比例: {sample_ratio:.1%})...")
    print(f"📐 输入位姿格式: {pose_format.upper()} ({'Camera-to-World' if pose_format == 'c2w' else 'World-to-Camera'})")
    if depth_scale != 1.0:
        print(f"🔧 深度和位姿缩放因子: {depth_scale}x")

    # 加载外部深度数据（如果提供）
    external_depths = None
    if external_depth_npz:
        print(f"📁 加载外部深度文件: {external_depth_npz}")
        depth_data = np.load(external_depth_npz)
        external_depths = depth_data['data']  # 形状应该是 (num_frames, H, W)
        print(f"   外部深度数据形状: {external_depths.shape}")
        print(f"   深度范围: [{external_depths.min():.3f}, {external_depths.max():.3f}] (mm)")
        # 将深度从毫米转换为米
        external_depths = external_depths / 1000.0
        print(f"   转换后深度范围: [{external_depths.min():.6f}, {external_depths.max():.6f}] (m)")

    # 初始化存储列表
    pts3ds_list = []      # 存储每帧的3D点云
    colors_list = []      # 存储每帧的颜色信息
    conf_list = []        # 存储每帧的置信度
    poses_list = []       # 存储每帧的相机位姿
    intrinsics_list = []  # 存储每帧的相机内参

    # 逐帧处理数据
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:  # 每10帧打印一次进度
            print(f"   处理第 {frame_idx} 帧...")

        # 构建文件路径
        # SCARED数据集的文件命名格式：000000.npz, 000000.npy, 000000.jpg
        cam_file = os.path.join(data_dir, "cam", f"{frame_idx:06d}.npz")      # 相机参数文件
        rgb_file = os.path.join(data_dir, "rgb", f"{frame_idx:06d}.jpg")      # RGB图像文件

        # 根据是否使用外部深度文件来决定深度文件路径
        if external_depths is not None:
            # 使用外部深度数据
            if frame_idx >= external_depths.shape[0]:
                print(f"     跳过第 {frame_idx} 帧 - 超出外部深度数据范围")
                continue
            depth_file_exists = True  # 外部深度数据存在
        else:
            # 使用原始深度文件
            depth_file = os.path.join(data_dir, "depth", f"{frame_idx:06d}.npy")  # 深度图文件
            depth_file_exists = os.path.exists(depth_file)

        # 检查必需文件是否存在
        required_files = [cam_file, rgb_file]
        if external_depths is None:  # 修复：使用 is None 而不是 not external_depths
            required_files.append(depth_file)

        if not all(os.path.exists(f) for f in required_files):
            print(f"     跳过第 {frame_idx} 帧 - 文件不存在")
            continue

        # 加载数据文件
        cam_data = np.load(cam_file)           # 加载相机参数（.npz格式）
        rgb_img = np.array(Image.open(rgb_file))  # 加载RGB图像

        # 加载深度数据
        if external_depths is not None:
            # 使用外部深度数据
            depth = external_depths[frame_idx]  # 从外部npz文件中获取对应帧的深度
            print(f"     使用外部深度数据第 {frame_idx} 帧")
        else:
            # 使用原始深度文件
            depth = np.load(depth_file)            # 加载深度图（.npy格式）

        # 从相机参数文件中提取位姿和内参
        pose = cam_data['pose'].copy()    # 4×4相机位姿矩阵（cam2world格式）
        intrinsics = cam_data['intrinsics']  # 3×3相机内参矩阵

        # 将位移向量从毫米转换为米（位姿矩阵的最后一列前3个元素）
        #pose[:3, 3] = pose[:3, 3] / 1000.0
        if frame_idx == 0:  # 只在第一帧打印转换信息
            print(f"     位移向量单位转换: mm -> m")
            print(f"     转换后位移: [{pose[0,3]:.6f}, {pose[1,3]:.6f}, {pose[2,3]:.6f}] (m)")

        # === 应用深度缩放 ===
        # 如果指定了缩放因子，对深度数据进行缩放
        if depth_scale != 1.0:
            depth = depth * depth_scale
            if frame_idx == 0:  # 只在第一帧打印缩放信息
                print(f"     深度缩放 {depth_scale}x: 范围变为 [{depth.min():.6f}, {depth.max():.6f}] (m)")

        # 调整RGB图像尺寸以匹配深度图
        # 有时RGB图像和深度图的分辨率不一致，需要调整RGB图像尺寸
        if depth.shape[:2] != rgb_img.shape[:2]:
            # 将RGB图像resize到与深度图相同的尺寸
            rgb_img = np.array(Image.fromarray(rgb_img).resize((depth.shape[1], depth.shape[0])))

        # === 创建3D点云（相机坐标系） ===
        # 这一步将2D深度图转换为3D点云
        H, W = depth.shape  # 获取深度图的高度和宽度

        # 创建像素坐标网格
        # y对应行（高度方向），x对应列（宽度方向）
        y, x = np.mgrid[0:H, 0:W]  # 生成H×W的坐标网格

        # 将像素坐标转换为归一化相机坐标
        # 使用相机内参矩阵进行去畸变和归一化
        # intrinsics[0,2]和intrinsics[1,2]是主点坐标(cx, cy)
        # intrinsics[0,0]和intrinsics[1,1]是焦距(fx, fy)
        x_norm = (x - intrinsics[0, 2]) / intrinsics[0, 0]  # 归一化x坐标
        y_norm = (y - intrinsics[1, 2]) / intrinsics[1, 1]  # 归一化y坐标

        # 根据针孔相机模型，将深度值和归一化坐标结合得到3D点
        # 相机坐标系：X轴向右，Y轴向下，Z轴向前（深度方向）
        pts3d_cam = np.stack([
            x_norm * depth,  # X = (u - cx) * Z / fx
            y_norm * depth,  # Y = (v - cy) * Z / fy
            depth           # Z = depth
        ], axis=-1)  # 形状为(H, W, 3)

        # === 坐标系变换：相机坐标系 -> 世界坐标系 ===
        # 根据输入位姿格式处理变换矩阵
        if pose_format == 'c2w':
            # 输入已经是cam2world格式，直接使用
            cam2world = pose
            if frame_idx == 0:  # 只在第一帧打印信息
                print(f"     使用C2W格式位姿，直接应用变换")
        elif pose_format == 'w2c':
            # 输入是world2cam格式，需要取逆得到cam2world
            cam2world = np.linalg.inv(pose)
            if frame_idx == 0:  # 只在第一帧打印信息
                print(f"     输入W2C格式位姿，取逆转换为C2W格式")
        else:
            raise ValueError(f"不支持的位姿格式: {pose_format}，请使用 'c2w' 或 'w2c'")

        # === 应用位姿缩放 ===
        # 如果指定了缩放因子，对cam2world的位移向量进行相同的缩放
        # 这样保持深度和位姿的一致性
        if depth_scale != 1.0:
            cam2world[:3, 3] = cam2world[:3, 3] * depth_scale  # 只缩放位移向量T，不缩放旋转矩阵R
            if frame_idx == 0:  # 只在第一帧打印缩放信息
                print(f"     位姿位移缩放 {depth_scale}x: T向量变为 {cam2world[:3, 3]}")

        # 将相机坐标系的点转换到世界坐标系
        H, W = pts3d_cam.shape[:2]
        pts3d_cam_flat = pts3d_cam.reshape(-1, 3)  # 展平为(H*W, 3)

        # 添加齐次坐标（第4维设为1）以便进行4×4矩阵变换
        pts3d_cam_homo = np.hstack([pts3d_cam_flat, np.ones((pts3d_cam_flat.shape[0], 1))])

        # 应用cam2world变换矩阵
        # 矩阵乘法：(4×4) × (4×H*W) -> (4×H*W)，然后转置得到(H*W×4)
        pts3d_world_homo = (cam2world @ pts3d_cam_homo.T).T
        pts3d_world_flat = pts3d_world_homo[:, :3]  # 取前3维，去掉齐次坐标

        # 重新整形回原始的图像形状
        pts3d = pts3d_world_flat.reshape(H, W, 3)  # 恢复为(H, W, 3)形状

        # === 创建置信度和颜色数据 ===
        # 置信度：有效深度值的像素置信度为1，无效深度值为0
        conf = (depth > 0).astype(np.float32)  # 深度>0的像素认为是有效的

        # 颜色归一化：将RGB值从[0,255]范围归一化到[0,1]范围
        colors = rgb_img.astype(np.float32) / 255.0

        # === 点云采样优化 ===
        # 由于原始点云密度很高（H×W个点），直接可视化会很慢
        # 根据sample_ratio计算合适的采样步长
        # 例如：sample_ratio=0.05意味着保留5%的点，步长约为sqrt(1/0.05)≈4.5
        step = max(1, int(1.0 / np.sqrt(sample_ratio)))

        # 调用均匀采样函数减少点云密度
        sampled_pts, sampled_colors, sampled_conf = sample_pointcloud_uniform(
            pts3d, colors, conf, step
        )

        # === 转换为PyTorch张量 ===
        # CUT3R的可视化工具需要torch.Tensor格式的数据
        pts3ds_list.append(torch.from_numpy(sampled_pts).float())    # 3D点坐标
        colors_list.append(torch.from_numpy(sampled_colors).float()) # RGB颜色
        conf_list.append(torch.from_numpy(sampled_conf).float())     # 置信度

        # 保存处理后的位姿和内参（numpy格式）
        poses_list.append(cam2world)      # 保存cam2world位姿矩阵（已经变换和缩放过的）
        intrinsics_list.append(intrinsics)  # 相机内参矩阵

    print(f"   成功加载 {len(pts3ds_list)} 帧")
    return pts3ds_list, colors_list, conf_list, poses_list, intrinsics_list

def run_optimized_visualization(data_dir, num_frames=50, sample_ratio=0.05, port=8087, external_depth_npz=None, depth_scale=1.0, pose_format='c2w'):
    """
    运行优化的可视化

    这是主要的可视化函数，整合了数据加载、处理和可视化的完整流程。

    主要步骤：
    1. 加载多帧数据（RGB、深度、相机参数）
    2. 将点云转换到世界坐标系
    3. 处理相机位姿（支持c2w和w2c格式）
    4. 应用位姿缩放和调整
    5. 启动交互式3D可视化

    参数说明：
    - data_dir: SCARED数据集目录
    - num_frames: 要可视化的帧数
    - sample_ratio: 点云采样比例（减少点数以提高性能）
    - port: web服务器端口
    - external_depth_npz: 外部深度npz文件路径（可选，用于替换原始深度数据）
    - depth_scale: 深度和位姿缩放因子（可选，默认1.0不缩放）
    - pose_format: 输入位姿格式，'c2w'或'w2c'（默认'c2w'）

    修正版本特点：
    - 支持c2w和w2c两种位姿格式输入
    - 自动处理位姿格式转换
    - 将点云从相机坐标系转换到世界坐标系
    - 确保多帧点云正确对齐形成连续表面
    - 使用原始相机位姿，不进行人工缩放或调整
    """
    print("⚡ 启动优化可视化 (修正版)...")
    print("=" * 50)
    print(f"   帧数: {num_frames}")
    print(f"   采样比例: {sample_ratio:.1%}")
    print(f"   端口: {port}")
    print(f"   � 位姿格式: {pose_format.upper()} ({'Camera-to-World' if pose_format == 'c2w' else 'World-to-Camera'})")
    if depth_scale != 1.0:
        print(f"   📏 深度和位姿缩放: {depth_scale}x")
    else:
        print(f"   📍 使用原始深度和位姿，无缩放")
    if external_depth_npz:
        print(f"   📁 使用外部深度文件: {external_depth_npz}")
    else:
        print(f"   📁 使用原始深度文件")
    print("=" * 50)

    # === 第1步：加载数据 ===
    pts3ds_list, colors_list, conf_list, poses_list, intrinsics_list = load_optimized_data(
        data_dir, num_frames, sample_ratio, external_depth_npz, depth_scale, pose_format
    )

    if not pts3ds_list:
        print("❌ 没有成功加载任何帧")
        return

    # === 第2步：准备可视化数据格式 ===
    # 将PyTorch张量转换为numpy数组，因为可视化工具需要numpy格式
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_list]    # 3D点云数据
    colors_to_vis = [c.cpu().numpy() for c in colors_list]    # 颜色数据
    conf_to_vis = [[c.cpu().numpy()] for c in conf_list]      # 置信度数据（需要嵌套列表格式）
    edge_colors = [None] * len(pts3ds_to_vis)                 # 边缘颜色（设为None使用默认）

    # === 第3步：提取相机位置和朝向 ===
    # poses_list中现在直接存储的是cam2world变换矩阵（已经在数据加载时处理过）
    # 无需再次进行逆变换，直接提取位置和朝向即可
    # cam2world[:3, 3]是相机在世界坐标系中的位置（平移向量）
    # cam2world[:3, :3]是相机在世界坐标系中的朝向（旋转矩阵）
    positions = np.array([pose[:3, 3] for pose in poses_list])  # 相机位置（已经是cam2world格式）
    rotations = np.array([pose[:3, :3] for pose in poses_list]) # 相机朝向（已经是cam2world格式）

    # 打印相机轨迹的统计信息
    print(f"   相机位置范围 (世界坐标系):")
    print(f"     X: {positions[:, 0].min():.6f} 到 {positions[:, 0].max():.6f}")
    print(f"     Y: {positions[:, 1].min():.6f} 到 {positions[:, 1].max():.6f}")
    print(f"     Z: {positions[:, 2].min():.6f} 到 {positions[:, 2].max():.6f}")

    # === 第4步：使用处理后的相机位置 ===
    # 直接使用从cam2world矩阵提取的相机位置（已经包含了缩放处理）
    # 这样可以保持数据的真实性，便于调试和验证
    scaled_positions = positions  # 直接使用已处理的位置

    # === 第5步：构建相机字典 ===
    # 这是CUT3R可视化工具需要的相机参数格式
    cam_dict = {
        'focal': np.array([intr[0, 0] for intr in intrinsics_list]),  # 焦距列表
        'pp': np.array([[intr[0, 2], intr[1, 2]] for intr in intrinsics_list]),  # 主点坐标列表
        'R': rotations,      # 旋转矩阵列表（相机朝向）
        't': scaled_positions,  # 位置向量列表（已处理的相机位置）
    }

    # === 第6步：统计信息计算 ===
    # 计算优化后的数据统计信息
    total_points = sum(p.size // 3 for p in pts3ds_to_vis)  # 总点数（每个点有3个坐标）
    avg_points_per_frame = total_points // len(pts3ds_to_vis)  # 平均每帧点数

    print(f"📊 优化后统计:")
    print(f"   帧数: {len(pts3ds_to_vis)}")
    print(f"   总点数: {total_points:,}")
    print(f"   平均每帧点数: {avg_points_per_frame:,}")
    print(f"   相机数量: {len(cam_dict['focal'])}")

    # 分析相机轨迹范围
    position_range = np.max(positions, axis=0) - np.min(positions, axis=0)  # 相机位置范围
    print(f"   相机位移范围: {position_range}")

    # 计算轨迹总长度
    total_trajectory_length = 0
    for i in range(1, len(positions)):
        total_trajectory_length += np.linalg.norm(positions[i] - positions[i-1])
    print(f"   轨迹总长度: {total_trajectory_length:.6f} m")

    # === 第7步：创建可视化对象 ===
    # 创建一个虚拟模型类，因为PointCloudViewer需要一个模型参数
    class DummyModel:
        """虚拟模型类，用于满足PointCloudViewer的接口要求"""
        pass

    # === 第8步：启动可视化 ===
    try:
        # 创建PointCloudViewer实例
        # 这是CUT3R提供的交互式3D可视化工具
        viewer = PointCloudViewer(
            DummyModel(),           # 虚拟模型对象
            None,                   # 不需要额外的模型参数
            pts3ds_to_vis,         # 3D点云数据列表
            colors_to_vis,         # 颜色数据列表
            conf_to_vis,           # 置信度数据列表
            cam_dict,              # 相机参数字典
            device="cpu",          # 使用CPU进行计算
            edge_color_list=edge_colors,  # 边缘颜色（None表示使用默认）
            show_camera=True,      # 显示相机位置和朝向
            vis_threshold=0.1,     # 可视化阈值（置信度低于此值的点不显示）
            size=256,              # 渲染尺寸
            port=port              # web服务器端口
        )

        # 打印成功信息和使用说明
        print("🎉 优化可视化启动成功!")
        print(f"📱 访问 http://localhost:{port}")
        print("🎮 优化特性:")
        print(f"   - 每帧采样到 {sample_ratio:.1%} 的点数")
        print("   - 使用原始相机位姿，无人工调整")
        print("   - 点云密度降低但保持整体结构")
        print("   - 真实反映数据中的相机运动")
        print("=" * 50)

        # 启动可视化服务器（这会阻塞程序直到用户关闭）
        viewer.run()

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误信息用于调试

def main():
    """
    主函数：解析命令行参数并启动可视化

    这个函数处理命令行参数，让用户可以自定义可视化的各种参数。
    支持的参数包括：
    - data_dir: SCARED数据集目录路径
    - num_frames: 要可视化的帧数
    - sample_ratio: 点云采样比例（用于性能优化）
    - port: web服务器端口号
    - external_depth_npz: 外部深度npz文件路径（可选）
    - depth_scale: 深度和位姿缩放因子（可选）
    - pose_format: 输入位姿格式，'c2w'或'w2c'（可选，默认'c2w'）
    """
    parser = argparse.ArgumentParser(description="优化多帧可视化")

    # 数据目录参数
    parser.add_argument("--data_dir", type=str,
                       default="/hy-tmp/CUT3R/processed_scared_split_newV3/train/dataset7_keyframe31",
                       help="SCARED数据集目录路径")

    # 可视化参数
    parser.add_argument("--num_frames", type=int, default=50,
                       help="要可视化的帧数")
    parser.add_argument("--sample_ratio", type=float, default=0.05,
                       help="点云采样比例 (0.05 = 5%)")
    parser.add_argument("--port", type=int, default=8087,
                       help="web服务器端口号")

    # 深度数据选项
    parser.add_argument("--external_depth_npz", type=str, default=None,
                       help="外部深度npz文件路径（可选，用于替换原始深度数据）")
    parser.add_argument("--depth_scale", type=float, default=1.0,
                       help="深度和位姿缩放因子（可选，默认1.0不缩放，建议20.0）")

    # 位姿格式选项
    parser.add_argument("--pose_format", type=str, default='c2w', choices=['c2w', 'w2c'],
                       help="输入位姿格式：'c2w'表示camera-to-world（默认），'w2c'表示world-to-camera")

    args = parser.parse_args()

    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return

    # 检查外部深度文件是否存在（如果提供）
    if args.external_depth_npz and not os.path.exists(args.external_depth_npz):
        print(f"❌ 外部深度文件不存在: {args.external_depth_npz}")
        return

    # 启动可视化
    run_optimized_visualization(
        args.data_dir,              # 数据目录
        args.num_frames,            # 帧数
        args.sample_ratio,          # 采样比例
        args.port,                  # 端口
        args.external_depth_npz,    # 外部深度文件
        args.depth_scale,           # 深度和位姿缩放因子
        args.pose_format            # 位姿格式
    )

if __name__ == "__main__":
    """
    程序入口点

    当直接运行这个脚本时（而不是作为模块导入），会执行main()函数。
    这是Python的标准做法，确保脚本既可以独立运行，也可以作为模块被其他代码导入。
    """
    main()
