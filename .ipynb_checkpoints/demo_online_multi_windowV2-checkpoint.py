#!/usr/bin/env python3
"""
多窗口3D点云推理和可视化脚本 - 锚点帧对齐版本 (V2)
这个脚本实现了使用锚点帧对齐机制的CUT3R推理，主要功能：
1. 将长序列分割成多个窗口，相邻窗口共享一个锚点帧
2. 每个窗口独立推理，重置模型状态
3. 通过锚点帧在两个窗口中的位姿差异进行对齐
4. 避免长序列导致的内存溢出和性能下降

核心改进（相比V1）：
- 使用CUT3R自身的匹配能力，而非外部Procrustes分析
- 锚点帧提供天然的几何约束，对齐更准确
- 直接基于位姿变换，避免多帧匹配的累积误差
"""
 
import sys; sys.path.insert(0, 'src')
import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
from scipy.spatial.transform import Rotation
 
# 设置随机种子确保结果可重现
random.seed(42)
 
 
def parse_args():
    """
    解析命令行参数
    
    定义了脚本所需的所有参数：
    - 模型路径、输入序列路径、设备等基本参数
    - 窗口大小等滑动窗口相关参数（移除了overlap_size，因为使用锚点帧）
    """
    parser = argparse.ArgumentParser(
        description="使用锚点帧对齐的多窗口3D点云推理"
    )
    
    # 基本模型参数
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="预训练模型检查点的路径",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="包含图像序列的目录路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行推理的设备 (例如 'cuda' 或 'cpu')",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="224",
        help="输入图像将被缩放到的尺寸; 如果使用224+linear模型选择224，否则选择512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="点云查看器的可视化阈值，范围从1到无穷大",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp_multi_window_v2",
        help="输出目录路径",
    )
    
    # 滑动窗口相关参数
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="每个窗口的帧数",
    )
 
    return parser.parse_args()
 
 
def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
):
    """
    为推理准备输入视图数据
    
    这个函数将图像路径列表转换为CUT3R模型可以处理的输入格式。
    每个视图包含图像数据和相关的元数据。
    
    Args:
        img_paths: 图像文件路径列表
        img_mask: 指示有效图像的布尔标志列表
        size: 目标图像尺寸
        raymaps: 射线图列表（可选）
        raymap_mask: 指示有效射线图的标志（可选）
        revisit: 每个视图重访问的次数
        update: 是否在重访问时更新状态
        
    Returns:
        list: 视图字典列表，每个字典包含模型所需的所有数据
    """
    # 延迟导入图像加载器（在添加ckpt路径后需要）
    from src.dust3r.utils.image import load_images
 
    # 加载并预处理所有图像
    images = load_images(img_paths, size=size)
    views = []
 
    if raymaps is None and raymap_mask is None:
        # 只提供图像的情况（最常见的情况）
        for i in range(len(images)):
            # 为每个图像创建一个视图字典
            view = {
                "img": images[i]["img"],  # 图像张量
                "ray_map": torch.full(    # 创建空的射线图（填充NaN）
                    (
                        images[i]["img"].shape[0],  # batch size
                        6,                          # 射线图通道数
                        images[i]["img"].shape[-2], # 高度
                        images[i]["img"].shape[-1], # 宽度
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),  # 原始图像尺寸
                "idx": i,                    # 视图索引
                "instance": str(i),          # 实例标识符
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),  # 初始相机姿态（单位矩阵）
                "img_mask": torch.tensor(True).unsqueeze(0),     # 图像掩码（有效）
                "ray_mask": torch.tensor(False).unsqueeze(0),    # 射线图掩码（无效）
                "update": torch.tensor(True).unsqueeze(0),       # 更新标志
                "reset": torch.tensor(False).unsqueeze(0),       # 重置标志
            }
            views.append(view)
    else:
        # 同时处理图像和射线图的情况（更复杂的场景）
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)
 
        j = 0  # 图像索引
        k = 0  # 射线图索引
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)
 
    # 处理重访问情况（用于某些特殊的推理策略）
    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views
 
    return views
 
 
def compute_anchor_alignment(pose1_last, pose2_first):
    """
    基于锚点帧计算窗口间的对齐变换
    
    这是V2版本的核心改进：使用锚点帧在两个窗口中的位姿差异
    来计算对齐变换，而非V1中的多帧Procrustes分析。
    
    原理：
    - 同一帧在窗口1中的位姿为pose1_last
    - 同一帧在窗口2中的位姿为pose2_first（通常是单位矩阵）
    - 计算变换T，使得pose2_first变换后等于pose1_last
    
    Args:
        pose1_last: (4, 4) 锚点帧在第一个窗口中的位姿
        pose2_first: (4, 4) 锚点帧在第二个窗口中的位姿
        
    Returns:
        T: (4, 4) 变换矩阵
    """
    # 方法A：直接矩阵运算
    # T = pose1_last @ np.linalg.inv(pose2_first)
    T = pose1_last @ np.linalg.inv(pose2_first)
    
    return T
 
 
def apply_transformation_to_poses(poses, T):
    """
    将4x4变换矩阵应用到相机姿态上
    
    对每个相机姿态应用计算出的对齐变换，使当前窗口的姿态
    与前一个窗口保持一致。
    
    Args:
        poses: (N, 4, 4) 原始相机姿态
        T: (4, 4) 变换矩阵
        
    Returns:
        transformed_poses: (N, 4, 4) 变换后的相机姿态
    """
    transformed_poses = poses.copy()
    
    # 为每个姿态应用变换
    for i in range(len(poses)):
        # 应用变换：T_new = T * T_old
        # 这个操作将当前姿态变换到与前一窗口对齐的坐标系中
        transformed_poses[i] = T @ poses[i]
    
    return transformed_poses
 
 
def create_sliding_windows_with_anchor(total_frames, window_size):
    """
    创建带锚点帧的滑动窗口索引
    
    与V1不同，V2版本使用锚点帧而非重叠帧：
    - 第一个窗口：[0, 1, 2, ..., window_size-1]
    - 第二个窗口：[window_size-1, window_size, ..., 2*window_size-1] (锚点帧: window_size-1)
    - 第三个窗口：[2*window_size-1, 2*window_size, ..., 3*window_size-1] (锚点帧: 2*window_size-1)
    
    Args:
        total_frames: 总帧数
        window_size: 每个窗口的帧数
        
    Returns:
        List of (start_idx, end_idx, anchor_idx) tuples: 窗口起始、结束索引和锚点帧索引
    """
    windows = []
    start_idx = 0
    
    while start_idx < total_frames:
        # 计算当前窗口的结束索引
        end_idx = min(start_idx + window_size, total_frames)
        
        # 确定锚点帧索引（除第一个窗口外，锚点帧是前一个窗口的最后一帧）
        if start_idx == 0:
            # 第一个窗口没有锚点帧
            anchor_idx = None
        else:
            # 后续窗口的锚点帧是该窗口的第一帧（也是前一窗口的最后一帧）
            anchor_idx = start_idx
            
        windows.append((start_idx, end_idx, anchor_idx))
        
        # 如果已经处理完所有帧，退出循环
        if end_idx >= total_frames:
            break
            
        # 下一个窗口的起始位置：当前结束位置减1（重用最后一帧作为锚点）
        start_idx = end_idx - 1
    
    return windows
 
 
def parse_seq_path(p):
    """
    解析序列路径，支持目录和视频文件
    
    处理两种输入：
    1. 目录：直接读取目录中的图像文件
    2. 视频文件：提取视频帧到临时目录
    
    Args:
        p: 输入路径（目录或视频文件）
        
    Returns:
        img_paths: 图像路径列表
        tmpdirname: 临时目录名（如果创建了的话）
    """
    if os.path.isdir(p):
        # 处理目录输入：直接列出所有文件并排序
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        # 处理视频文件输入：提取帧到临时目录
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件 {p}")
        
        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"错误：视频FPS为0 {p}")
        
        # 设置帧采样间隔（这里设为1，即不跳帧）
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - 视频FPS: {video_fps}, 帧间隔: {frame_interval}, 总读取帧数: {len(frame_indices)}"
        )
        
        # 创建临时目录并提取帧
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # 定位到指定帧
            ret, frame = cap.read()
            if not ret:
                break
            # 保存帧到临时目录
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    
    return img_paths, tmpdirname
 
 
def run_inference_multi_window_v2(args):
    """
    执行多窗口推理流水线，使用锚点帧对齐（V2版本）
    
    这是V2版本的核心函数，实现了改进的多窗口推理流程：
    1. 准备数据和模型
    2. 创建带锚点帧的滑动窗口
    3. 逐窗口推理
    4. 锚点帧对齐
    5. 结果合并和可视化
    """
    # 设置计算设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，切换到CPU")
        device = "cpu"
 
    # 添加检查点路径（dust3r包中的模型导入所需）
    add_path_to_dust3r(args.model_path)
 
    # 延迟导入模型和推理函数（添加ckpt路径后）
    from src.dust3r.inference import inference_recurrent
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
 
    # 准备图像文件路径
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"在 {args.seq_path} 中没有找到图像，请检验路径")
        return
 
    print(f"在 {args.seq_path} 中找到 {len(img_paths)} 张图像")
    
    # 创建带锚点帧的滑动窗口
    windows = create_sliding_windows_with_anchor(len(img_paths), args.window_size)
    print(f"创建了 {len(windows)} 个窗口，窗口大小 {args.window_size}，使用锚点帧对齐")
    
    # 清理临时目录
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)
 
    # 加载和准备模型
    print(f"从 {args.model_path} 加载模型...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()  # 设置为评估模式
 
    # 处理每个窗口
    all_outputs = []           # 存储所有窗口的输出
    all_poses = []             # 存储所有窗口的相机姿态
    prev_window_poses = None   # 前一个窗口的姿态（用于对齐）
    
    total_start_time = time.time()
    
    # 逐个处理每个窗口
    for window_idx, (start_idx, end_idx, anchor_idx) in enumerate(windows):
        # 获取当前窗口的图像路径
        window_img_paths = img_paths[start_idx:end_idx]
        window_img_mask = [True] * len(window_img_paths)  # 所有图像都是有效的
        
        print(f"\n处理窗口 {window_idx + 1}/{len(windows)}: 帧 {start_idx}-{end_idx-1}")
        if anchor_idx is not None:
            print(f"锚点帧: {anchor_idx}")
        
        # 为当前窗口准备输入视图
        views = prepare_input(
            img_paths=window_img_paths,
            img_mask=window_img_mask,
            size=args.size,
            revisit=1,
            update=True,
        )
        
        # 运行当前窗口的推理（使用全新的状态）
        window_start_time = time.time()
        outputs, state_args = inference_recurrent(views, model, device)
        window_time = time.time() - window_start_time
        
        print(f"窗口 {window_idx + 1} 推理完成，用时 {window_time:.2f} 秒")
        
        # 从当前窗口提取相机姿态
        pred_poses = []
        for pred in outputs["pred"]:
            pose_encoding = pred["camera_pose"]  # 获取姿态编码
            # 将姿态编码转换为4x4变换矩阵
            camera_pose = pose_encoding_to_camera(pose_encoding).cpu().numpy()
            pred_poses.append(camera_pose[0])  # 移除batch维度
        
        pred_poses = np.array(pred_poses)  # 转换为numpy数组 (N, 4, 4)
        
        # 如果不是第一个窗口，进行锚点帧对齐
        if window_idx > 0 and prev_window_poses is not None and anchor_idx is not None:
            # 获取锚点帧的位姿
            # 在前一个窗口中，锚点帧是最后一帧
            pose1_last = prev_window_poses[-1]  # 前一窗口的最后一帧
            
            # 在当前窗口中，锚点帧是第一帧  
            pose2_first = pred_poses[0]  # 当前窗口的第一帧
            
            print(f"锚点帧对齐: 前窗口最后帧 -> 当前窗口第一帧")
            
            # 计算对齐变换
            T = compute_anchor_alignment(pose1_last, pose2_first)
            
            # 将变换应用到当前窗口的所有姿态
            pred_poses = apply_transformation_to_poses(pred_poses, T)
            
            # 输出对齐信息（用于调试）
            print(f"应用对齐变换: T_norm={np.linalg.norm(T - np.eye(4)):.4f}")
        
        # 存储结果
        all_outputs.append(outputs)
        all_poses.append(pred_poses)
        prev_window_poses = pred_poses  # 更新前一窗口姿态
        
    # 计算总体性能统计
    total_time = time.time() - total_start_time
    total_frames = sum(end_idx - start_idx for start_idx, end_idx, _ in windows)
    avg_fps = total_frames / total_time
    
    print(f"\n总处理完成，用时 {total_time:.2f} 秒")
    print(f"平均FPS: {avg_fps:.2f}")
    print(f"总处理帧数: {total_frames}")
    
    # 合并所有输出（去除锚点帧重复）
    print("合并窗口输出（移除锚点帧重复）...")
    merged_pred = []
    merged_views = []
    
    for window_idx, window_outputs in enumerate(all_outputs):
        if window_idx == 0:
            # 第一个窗口：取所有帧
            merged_pred.extend(window_outputs["pred"])
            merged_views.extend(window_outputs["views"])
        else:
            # 后续窗口：跳过第一帧（锚点帧重复），只取新帧
            merged_pred.extend(window_outputs["pred"][1:])
            merged_views.extend(window_outputs["views"][1:])
    
    print(f"合并后的唯一帧数: {len(merged_pred)}")
    
    # 创建合并的输出结构
    merged_outputs = {
        "pred": merged_pred,
        "views": merged_views
    }
    
    # 使用原始的prepare_output函数保存结果
    from demo_online import prepare_output
    
    print("准备输出用于可视化...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        merged_outputs, args.output_dir, 1, True
    )
    
    print(f"结果保存到 {args.output_dir}")
    
    # 添加像原始demo_online.py中的可视化
    print("启动点云查看器...")
    try:
        from viser_utils import PointCloudViewer
        
        # 将张量转换为numpy数组用于可视化
        pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
        colors_to_vis = [c.cpu().numpy() for c in colors]
        edge_colors = [None] * len(pts3ds_to_vis)
 
        # 创建并运行点云查看器
        viewer = PointCloudViewer(
            model,
            None,  # static visualization不需要state_args
            pts3ds_to_vis,
            colors_to_vis,
            conf,
            cam_dict,
            device=device,
            edge_color_list=edge_colors,
            show_camera=True,
            vis_threshold=args.vis_threshold,
            size=args.size
        )
        viewer.run()
    except ImportError:
        print("viser_utils不可用，跳过可视化")
    
    print("锚点帧对齐的多窗口推理成功完成！")
    
    return merged_outputs, all_poses
 
 
def main():
    """
    主函数：解析参数并运行多窗口推理
    """
    args = parse_args()
    if not args.seq_path:
        print(
            "未找到输入！如果你想交互式上传输入，请使用我们的gradio演示"
        )
        return
    else:
        run_inference_multi_window_v2(args)
 
 
if __name__ == "__main__":
    main()
 
 
"""
========== V2版本核心改进总结 ==========
1. 【锚点帧机制】
   - 相邻窗口共享一个锚点帧
   - 利用CUT3R自身的匹配能力进行对齐
   
2. 【窗口划分策略】
   - 窗口1: [0, 1, 2, ..., 99] (100帧)
   - 窗口2: [99, 100, 101, ..., 199] (101帧，99为锚点)
   - 窗口3: [199, 200, 201, ..., 299] (101帧，199为锚点)
   
3. 【对齐算法】
   - 替换Procrustes分析为直接位姿变换
   - T = pose1_last @ pose2_first.inverse()
   - 基于同一帧在两个坐标系中的位姿差异
   
4. 【优势】
   - 更准确：使用CUT3R内在的几何约束
   - 更稳定：避免多帧匹配的累积误差
   - 更高效：直接矩阵运算，无需迭代优化
   
5. 【结果合并】
   - 第一个窗口：保留所有帧
   - 后续窗口：跳过锚点帧（第一帧），避免重复
   
6. 【预期效果】
   - 相比V1版本，对齐精度应该更高
   - 减少位姿漂移和累积误差
   - 保持全局3D结构的一致性
""" 
