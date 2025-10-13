#!/usr/bin/env python3
# =============================================================================
# Waymo开放数据集预处理代码详细解释版本
# 原始文件：preprocess_waymo.py
# =============================================================================

# 版权声明：Naver Corporation 2024，非商业用途许可
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# WayMo开放数据集的预处理代码
# 数据集地址：https://github.com/waymo-research/waymo-open-dataset
# 使用步骤：
# 1) 接受许可协议
# 2) 从Perception Dataset v1.4.2下载所有training/*.tfrecord文件
# 3) 将所有.tfrecord文件放在'/path/to/waymo_dir'中
# 4) 安装waymo_open_dataset包：
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) 执行脚本：`python preprocess_waymo.py --waymo_dir /path/to/waymo_dir`
# --------------------------------------------------------

# 系统库导入
import sys                    # 系统相关功能
import os                     # 操作系统接口
import os.path as osp         # 路径操作工具
import shutil                 # 高级文件操作
import json                   # JSON数据处理
from tqdm import tqdm         # 进度条显示
import PIL.Image              # Python图像库
import numpy as np            # 数值计算库

# 设置环境变量以启用OpenEXR支持（用于保存深度图）
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2                    # OpenCV计算机视觉库

# TensorFlow v1兼容模式（Waymo数据集使用TF格式）
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()   # 启用即时执行模式

# 导入项目内部模块
import path_to_root                                    # 添加项目根路径
from src.dust3r.utils.geometry import geotrf, inv     # 几何变换工具
from src.dust3r.utils.image import imread_cv2         # 图像读取工具
from src.dust3r.utils.parallel import parallel_processes as parallel_map  # 并行处理
from datasets_preprocess.utils import cropping        # 图像裁剪工具
from src.dust3r.viz import show_raw_pointcloud        # 点云可视化


def get_parser():
    """
    创建命令行参数解析器
    返回值：ArgumentParser对象
    """
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 必需参数：Waymo数据集目录路径（包含.tfrecord文件）
    parser.add_argument("--waymo_dir", required=True)
    
    # 必需参数：预计算的图像对文件路径
    parser.add_argument("--precomputed_pairs", required=True)
    
    # 可选参数：输出目录，默认为"data/waymo_processed"
    parser.add_argument("--output_dir", default="data/waymo_processed")
    
    # 可选参数：并行处理的工作进程数，默认为1
    parser.add_argument("--workers", type=int, default=1)
    
    return parser


def main(waymo_root, pairs_path, output_dir, workers=1):
    """
    主处理函数，完成Waymo数据集的完整预处理流程
    
    参数：
        waymo_root: 包含.tfrecord文件的Waymo数据集根目录
        pairs_path: 预计算的图像对文件路径
        output_dir: 处理后数据的输出目录
        workers: 并行处理的工作进程数
    """
    # 第一步：从tfrecord文件中提取原始帧数据
    extract_frames(waymo_root, output_dir, workers=workers)
    
    # 第二步：对提取的图像进行裁剪和缩放处理
    make_crops(output_dir, workers=workers)  # 注意：这里应该是workers而不是args.workers

    # 第三步：验证所有预计算的图像对是否都已成功处理
    with np.load(pairs_path) as data:
        scenes = data["scenes"]    # 场景名称列表
        frames = data["frames"]    # 帧名称列表
        pairs = data["pairs"]      # 图像对数组，格式为(场景ID, 图像1ID, 图像2ID)

    # 检查每个图像对中的所有图像是否存在
    for scene_id, im1_id, im2_id in pairs:
        for im_id in (im1_id, im2_id):
            # 构造图像文件路径
            path = osp.join(output_dir, scenes[scene_id], frames[im_id] + ".jpg")
            # 断言文件存在，如果不存在则提示错误信息
            assert osp.isfile(
                path
            ), f"Missing a file at {path=}\nDid you download all .tfrecord files?"

    # 清理临时目录
    shutil.rmtree(osp.join(output_dir, "tmp"))
    print("Done! all data generated at", output_dir)


def _list_sequences(db_root):
    """
    列出数据库根目录中的所有.tfrecord序列文件
    
    参数：
        db_root: 数据库根目录路径
    
    返回值：
        排序后的.tfrecord文件名列表
    """
    print(">> Looking for sequences in", db_root)
    
    # 查找所有以.tfrecord结尾的文件并排序
    res = sorted(f for f in os.listdir(db_root) if f.endswith(".tfrecord"))
    
    print(f"    found {len(res)} sequences")
    return res


def extract_frames(db_root, output_dir, workers=8):
    """
    从所有.tfrecord文件中并行提取帧数据
    
    参数：
        db_root: 包含.tfrecord文件的根目录
        output_dir: 输出目录
        workers: 并行工作进程数
    """
    # 获取所有序列文件列表
    sequences = _list_sequences(db_root)
    
    # 设置临时输出目录
    output_dir = osp.join(output_dir, "tmp")
    print(">> outputing result to", output_dir)
    
    # 为每个序列准备参数元组：(输入目录, 输出目录, 序列文件名)
    args = [(db_root, output_dir, seq) for seq in sequences]
    
    # 并行处理所有序列
    parallel_map(process_one_seq, args, star_args=True, workers=workers)


def process_one_seq(db_root, output_dir, seq):
    """
    处理单个.tfrecord序列文件
    
    参数：
        db_root: 输入数据库根目录
        output_dir: 输出目录
        seq: 序列文件名
    """
    # 为当前序列创建输出目录
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    
    # 相机标定文件路径
    calib_path = osp.join(out_dir, "calib.json")
    
    # 如果标定文件已存在，说明该序列已处理过，直接返回
    if osp.isfile(calib_path):
        return

    try:
        # 强制使用CPU处理（避免GPU内存问题）
        with tf.device("/CPU:0"):
            # 从序列文件中提取标定信息和帧数据
            calib, frames = extract_frames_one_seq(osp.join(db_root, seq))
    except RuntimeError:
        # 如果处理失败，输出错误信息并跳过
        print(f"/!\\ Error with sequence {seq} /!\\", file=sys.stderr)
        return  # 不保存任何数据

    # 遍历所有提取的帧
    for f, (frame_name, views) in enumerate(tqdm(frames, leave=False)):
        # 遍历当前帧的所有视图（不同摄像头）
        for cam_idx, view in views.items():
            # 从视图字典中弹出图像数据并转换为PIL图像
            img = PIL.Image.fromarray(view.pop("img"))
            
            # 保存图像，文件名格式：帧号(5位)_摄像头ID.jpg
            img.save(osp.join(out_dir, f"{f:05d}_{cam_idx}.jpg"))
            
            # 保存其他数据（位姿、像素坐标、3D点等）到npz文件
            np.savez(osp.join(out_dir, f"{f:05d}_{cam_idx}.npz"), **view)

    # 保存相机标定信息到JSON文件
    with open(calib_path, "w") as f:
        json.dump(calib, f)


def extract_frames_one_seq(filename):
    """
    从单个.tfrecord文件中提取所有帧数据
    
    参数：
        filename: .tfrecord文件的完整路径
    
    返回值：
        tuple: (相机标定信息, 帧数据列表)
    """
    # 导入Waymo数据集相关模块
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    print(">> Opening", filename)
    
    # 创建TensorFlow数据集读取器
    dataset = tf.data.TFRecordDataset(filename, compression_type="")

    calib = None    # 相机标定信息（只需读取一次）
    frames = []     # 存储所有帧数据

    # 遍历数据集中的每条记录
    for data in tqdm(dataset, leave=False):
        # 解析Waymo数据格式
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # 解析激光雷达和相机投影数据
        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}  # 存储当前帧的所有视图
        frames.append((frame.context.name, views))

        # 只在第一帧时读取相机标定信息
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append(
                    (
                        cam.name,                                  # 相机ID
                        dict(
                            width=cam.width,                       # 图像宽度
                            height=cam.height,                     # 图像高度
                            intrinsics=list(cam.intrinsic),        # 内参矩阵
                            extrinsics=list(cam.extrinsic.transform),  # 外参矩阵
                        ),
                    )
                )

        # 将激光雷达距离图像转换为3D点云
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose
        )

        # 合并所有3D点（在车辆坐标系中）
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # 将点云数据转换为TensorFlow张量，用于后续索引
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        # 处理当前帧的每个相机图像
        for i, image in enumerate(frame.images):
            # 选择与当前相机视图相关的3D点
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(
                tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32
            )

            # 提取相机位姿（4x4变换矩阵）
            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            # 解码JPEG图像
            rgb = tf.image.decode_jpeg(image.image).numpy()

            # 获取3D点在图像中的像素坐标
            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]

            # 存储当前视图的所有数据
            views[image.name] = dict(
                img=rgb,           # RGB图像
                pose=pose,         # 相机位姿
                pixels=pix,        # 像素坐标
                pts3d=pts3d,       # 对应的3D点
                timestamp=timestamp # 时间戳
            )

        # 可选：显示原始点云（调试用）
        if not "show full point cloud":  # 这个条件永远为False，所以不会执行
            show_raw_pointcloud(
                [v["pts3d"] for v in views.values()], 
                [v["img"] for v in views.values()]
            )

    return calib, frames


def make_crops(output_dir, workers=16, **kw):
    """
    对提取的图像进行裁剪和缩放处理
    
    参数：
        output_dir: 输出目录
        workers: 并行工作进程数
        **kw: 其他关键字参数
    """
    # 临时目录路径（包含原始提取的数据）
    tmp_dir = osp.join(output_dir, "tmp")
    
    # 获取所有序列目录
    sequences = _list_sequences(tmp_dir)
    
    # 为每个序列准备参数
    args = [(tmp_dir, output_dir, seq) for seq in sequences]
    
    # 并行处理所有序列的裁剪任务
    parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


def crop_one_seq(input_dir, output_dir, seq, resolution=512):
    """
    处理单个序列的图像裁剪和缩放
    
    参数：
        input_dir: 输入目录（临时目录）
        output_dir: 最终输出目录
        seq: 序列名称
        resolution: 目标分辨率
    """
    # 输入和输出目录路径
    seq_dir = osp.join(input_dir, seq)
    out_dir = osp.join(output_dir, seq)
    
    # 检查是否已处理过（通过检查特定文件是否存在）
    if osp.isfile(osp.join(out_dir, "00100_1.jpg")):
        return
    
    # 创建输出目录
    os.makedirs(out_dir, exist_ok=True)

    # 加载相机标定文件
    try:
        with open(osp.join(seq_dir, "calib.json")) as f:
            calib = json.load(f)
    except IOError:
        print(f"/!\\ Error: Missing calib.json in sequence {seq} /!\\", file=sys.stderr)
        return

    # 坐标轴变换矩阵（从Waymo坐标系转换到标准相机坐标系）
    # Waymo: x=前进方向, y=左侧, z=上方
    # 标准: x=右侧, y=下方, z=前进方向
    axes_transformation = np.array(
        [[0, -1, 0, 0],     # 新x = -旧y（右侧）
         [0, 0, -1, 0],     # 新y = -旧z（下方）
         [1, 0, 0, 0],      # 新z = 旧x（前进）
         [0, 0, 0, 1]]      # 齐次坐标
    )

    # 解析每个相机的标定参数
    cam_K = {}               # 内参矩阵
    cam_distortion = {}      # 畸变参数
    cam_res = {}             # 图像分辨率
    cam_to_car = {}          # 相机到车辆的变换矩阵
    
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        
        # 图像分辨率
        cam_res[cam_idx] = (W, H) = (cam_info["width"], cam_info["height"])
        
        # 解析内参：焦距、主点、畸变系数
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info["intrinsics"]
        
        # 构建内参矩阵
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        
        # 畸变参数
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        
        # 相机到车辆的变换矩阵
        cam_to_car[cam_idx] = np.asarray(cam_info["extrinsics"]).reshape(4, 4)

    # 获取所有帧文件（去掉扩展名.jpg的前三个字符）
    frames = sorted(f[:-4] for f in os.listdir(seq_dir) if f.endswith(".jpg"))

    # 可选：创建可视化对象（注释掉的代码）
    # from dust3r.viz import SceneViz
    # viz = SceneViz()

    # 处理每一帧
    for frame in tqdm(frames, leave=False):
        # 从文件名中提取相机索引（倒数第二个字符）
        cam_idx = frame[-1]  # 相机索引
        
        # 验证相机索引的有效性（Waymo有1-5号相机）
        assert cam_idx in "12345", f"bad {cam_idx=} in {frame=}"
        
        # 加载该帧的数据（位姿、像素坐标、3D点）
        data = np.load(osp.join(seq_dir, frame + ".npz"))
        car_to_world = data["pose"]    # 车辆到世界坐标的变换
        W, H = cam_res[cam_idx]        # 当前相机的分辨率

        # 处理深度图数据
        pos2d = data["pixels"].round().astype(np.uint16)  # 像素坐标
        x, y = pos2d.T                                    # 分离x,y坐标
        pts3d = data["pts3d"]                             # 3D点（车辆坐标系）
        
        # 将3D点从车辆坐标系转换到相机坐标系
        pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
        # 转换后：X=左右, Y=高度, Z=深度

        # 加载图像
        image = imread_cv2(osp.join(seq_dir, frame + ".jpg"))

        # 计算缩放分辨率（保持宽高比）
        output_resolution = (resolution, 1) if W > H else (1, resolution)
        
        # 对图像和内参进行缩放
        image, _, intrinsics2 = cropping.rescale_image_depthmap(
            image, None, cam_K[cam_idx], output_resolution
        )
        
        # 保存缩放后的图像（JPEG格式，质量80%）
        image.save(osp.join(out_dir, frame + ".jpg"), quality=80)

        # 创建深度图
        W, H = image.size
        depthmap = np.zeros((H, W), dtype=np.float32)
        
        # 将像素坐标转换到新的内参下
        pos2d = (
            geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d).round().astype(np.int16)
        )
        x, y = pos2d.T
        
        # 填充深度图（使用Z坐标作为深度值）
        depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
        
        # 保存深度图为EXR格式（浮点数格式，文件更小）
        cv2.imwrite(osp.join(out_dir, frame + ".exr"), depthmap)

        # 计算最终的相机到世界坐标变换
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        
        # 保存相机参数
        np.savez(
            osp.join(out_dir, frame + ".npz"),
            intrinsics=intrinsics2,                    # 缩放后的内参
            cam2world=cam2world,                       # 相机到世界的变换
            distortion=cam_distortion[cam_idx],        # 畸变参数
        )

        # 可选：添加到可视化（注释掉的代码）
        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    
    # 可选：显示可视化结果
    # viz.show()


# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = get_parser()
    args = parser.parse_args()
    
    # 执行主处理流程
    main(args.waymo_dir, args.precomputed_pairs, args.output_dir, workers=args.workers)


"""
=============================================================================
                              数据流程总结
=============================================================================

1. 数据输入：
   - 输入：Waymo .tfrecord 文件（包含图像、激光雷达、标定信息）
   - 格式：Protocol Buffer 二进制格式

2. 第一阶段处理（extract_frames）：
   - 解析 .tfrecord 文件
   - 提取 RGB 图像、3D 点云、相机位姿
   - 保存到临时目录：
     * XXXXX_Y.jpg（图像）
     * XXXXX_Y.npz（位姿、像素坐标、3D点）
     * calib.json（相机标定）

3. 第二阶段处理（make_crops）：
   - 图像缩放到指定分辨率
   - 生成深度图（从3D点云投影）
   - 坐标系转换（Waymo → 标准相机坐标系）
   - 最终输出：
     * XXXXX_Y.jpg（缩放后图像）
     * XXXXX_Y.exr（深度图）
     * XXXXX_Y.npz（相机参数）

4. 数据验证：
   - 检查预计算图像对是否完整
   - 清理临时文件

=============================================================================
                              关键技术要点
=============================================================================

1. 坐标系转换：
   - Waymo坐标系：x=前进, y=左侧, z=上方
   - 标准相机坐标系：x=右侧, y=下方, z=前进
   - 使用 4x4 变换矩阵进行转换

2. 深度图生成：
   - 从激光雷达3D点云投影到图像平面
   - 使用相机内参和外参进行投影
   - 保存为 EXR 格式（浮点数深度值）

3. 多相机处理：
   - Waymo 数据集包含 5 个相机（编号1-5）
   - 每个相机有独立的内参、外参、畸变参数
   - 并行处理提高效率

4. 文件命名规则：
   - 原始：帧号_相机号.扩展名
   - 例如：00123_2.jpg（第123帧，2号相机）

=============================================================================
""" 