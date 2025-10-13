"""
Point Odyssey 数据集读取代码详细解释
=====================================

本文件对 /hy-tmp/CUT3R/src/dust3r/datasets/pointodyssey.py 进行逐行详细解释
Point Odyssey 是一个用于跟踪和深度估计的多视角数据集

作者：AI Assistant
日期：2024
"""

# ==================== 导入模块部分 ====================

import os.path as osp
# os.path 模块用于处理文件路径，osp 是常用的别名
# 提供跨平台的路径操作功能，如 join(), dirname(), exists() 等

import cv2
# OpenCV 计算机视觉库，用于图像处理和计算机视觉任务
# 在这里主要用于图像的读取、处理和变换操作

import numpy as np
# NumPy 数组计算库，用于高效的数值计算
# 在深度学习和计算机视觉中广泛用于数组操作

import itertools
# Python 内置模块，提供迭代器工具
# 虽然在这个文件中导入了但实际没有使用

import os
# 操作系统接口模块，用于与操作系统交互
# 主要用于文件和目录操作，如 listdir(), join() 等

import sys
# 系统相关的参数和函数模块
# 在这里用于修改 Python 路径以导入自定义模块

# ==================== 路径设置部分 ====================

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
# 将当前文件的祖父目录添加到 Python 路径中
# 这样可以导入项目根目录下的模块
# __file__ 是当前文件的路径
# osp.dirname(__file__) 获取当前文件所在目录
# "..", ".." 表示向上两级目录

from tqdm import tqdm
# tqdm 是一个进度条库，用于显示循环和迭代的进度
# 在加载大量数据时提供可视化的进度反馈

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
# 导入基础的多视角数据集类
# PointOdyssey_Multi 类将继承这个基类
# BaseMultiViewDataset 提供了多视角数据集的通用功能

from dust3r.utils.image import imread_cv2
# 导入自定义的图像读取函数
# imread_cv2 是一个封装的图像读取函数，使用 OpenCV 后端

# ==================== Point Odyssey 数据集类定义 ====================

class PointOdyssey_Multi(BaseMultiViewDataset):
    """
    Point Odyssey 多视角数据集类
    
    Point Odyssey 是一个大规模的点跟踪数据集，包含：
    - 多视角RGB图像
    - 深度图
    - 相机参数（内参和外参）
    - 点轨迹标注
    
    该类负责加载和处理 Point Odyssey 数据集的数据
    """
    
    def __init__(self, *args, ROOT, **kwargs):
        """
        初始化 Point Odyssey 数据集
        
        参数:
            *args: 位置参数，传递给父类
            ROOT: 数据集根目录路径
            **kwargs: 关键字参数，传递给父类
        """
        
        self.ROOT = ROOT
        # 设置数据集根目录路径
        # ROOT 应该指向包含 train/test/val 子目录的主目录
        
        self.video = True
        # 标记这是视频数据集（时序数据）
        # 表示图像之间存在时间关系，可以用于视频相关的任务
        
        self.is_metric = True
        # 标记深度信息是度量深度（真实尺度）
        # True 表示深度值有真实的物理意义（如米为单位）
        # False 表示只有相对深度信息
        
        self.max_interval = 4
        # 设置最大时间间隔为4帧
        # 在选择图像序列时，相邻图像的最大时间间隔不超过4帧
        # 这有助于保持视觉连续性和跟踪的稳定性
        
        super().__init__(*args, **kwargs)
        # 调用父类的初始化方法
        # 父类 BaseMultiViewDataset 会处理通用的多视角数据集初始化
        
        assert self.split in ["train", "test", "val"]
        # 断言检查数据集分割必须是训练集、测试集或验证集之一
        # self.split 应该在父类初始化时设置
        
        # ==================== 场景列表定义 ====================
        self.scenes_to_use = [
            # 注释掉的场景（可能有问题或不使用）
            # 'cab_h_bench_3rd', 'cab_h_bench_ego1', 'cab_h_bench_ego2',
            
            # CNB 实验室场景
            "cnb_dlab_0215_3rd",    # CNB实验室2月15日第三人称视角
            "cnb_dlab_0215_ego1",   # CNB实验室2月15日自我中心视角1
            "cnb_dlab_0225_3rd",    # CNB实验室2月25日第三人称视角
            "cnb_dlab_0225_ego1",   # CNB实验室2月25日自我中心视角1
            
            # 舞蹈场景
            "dancing",              # 舞蹈场景
            "dancingroom0_3rd",     # 舞蹈房0第三人称视角
            
            # 脚部实验室场景
            "footlab_3rd",          # 脚部实验室第三人称视角
            "footlab_ego1",         # 脚部实验室自我中心视角1
            "footlab_ego2",         # 脚部实验室自我中心视角2
            
            # 人物场景
            "girl",                 # 女孩场景
            "girl_egocentric",      # 女孩自我中心视角
            "human_egocentric",     # 人类自我中心视角
            "human_in_scene",       # 场景中的人类
            "human_in_scene1",      # 场景中的人类1
            
            # KG 场景
            "kg",                   # KG场景
            "kg_ego1",              # KG自我中心视角1
            "kg_ego2",              # KG自我中心视角2
            
            # 厨房场景
            "kitchen_gfloor",       # 厨房地板场景
            "kitchen_gfloor_ego1",  # 厨房地板自我中心视角1
            "kitchen_gfloor_ego2",  # 厨房地板自我中心视角2
            
            # 桌子场景
            "scene_carb_h_tables",       # 碳氢桌子场景
            "scene_carb_h_tables_ego1",  # 碳氢桌子自我中心视角1
            "scene_carb_h_tables_ego2",  # 碳氢桌子自我中心视角2
            
            # J716 场景
            "scene_j716_3rd",       # J716第三人称视角
            "scene_j716_ego1",      # J716自我中心视角1
            "scene_j716_ego2",      # J716自我中心视角2
            
            # 录制场景
            "scene_recording_20210910_S05_S06_0_3rd",   # 2021年9月10日录制场景第三人称
            "scene_recording_20210910_S05_S06_0_ego2",  # 2021年9月10日录制场景自我中心2
            
            # 其他场景
            "scene1_0129",          # 1月29日场景1
            "scene1_0129_ego",      # 1月29日场景1自我中心视角
            
            # 研讨会场景
            "seminar_h52_3rd",      # H52研讨会第三人称视角
            "seminar_h52_ego1",     # H52研讨会自我中心视角1
            "seminar_h52_ego2",     # H52研讨会自我中心视角2
        ]
        # 这个列表定义了要使用的所有场景名称
        # Point Odyssey 数据集包含多个不同的场景，每个场景可能有不同视角
        # 第三人称视角 (_3rd) 和自我中心视角 (_ego) 提供了不同的观察角度
        
        self.loaded_data = self._load_data(self.split)
        # 调用数据加载方法，根据当前的数据集分割加载相应的数据
        # 返回值存储在 self.loaded_data 中（虽然这里没有使用返回值）

    def _load_data(self, split):
        """
        加载指定分割的数据
        
        这个方法扫描数据集目录，收集所有可用的图像和场景信息
        
        参数:
            split: 数据集分割名称 ("train", "test", "val")
        """
        
        root = os.path.join(self.ROOT, split)
        # 构建当前分割的根目录路径
        # 例如：/path/to/pointodyssey/train
        
        self.scenes = []
        # 初始化场景列表，存储实际使用的场景名称

        # ==================== 数据统计变量初始化 ====================
        offset = 0
        # 图像索引的偏移量，用于给每个图像分配全局唯一的ID
        
        scenes = []
        # 场景名称列表
        
        sceneids = []
        # 每个图像对应的场景ID列表
        
        scene_img_list = []
        # 每个场景包含的图像ID列表
        
        images = []
        # 所有图像的基础名称列表（不包含扩展名）
        
        start_img_ids = []
        # 可以作为起始图像的ID列表
        # 这些是可以开始多视角序列的图像

        j = 0  # 场景计数器

        # ==================== 遍历所有场景目录 ====================
        for scene in tqdm(os.listdir(root)):
            # 使用 tqdm 显示进度条，遍历根目录下的所有场景文件夹
            
            if scene not in self.scenes_to_use:
                continue
            # 如果当前场景不在预定义的使用列表中，跳过
            # 这样可以过滤掉不需要的场景
            
            scene_dir = osp.join(root, scene)
            # 构建当前场景的完整路径
            
            rgb_dir = osp.join(scene_dir, "rgb")
            # 构建RGB图像目录路径
            # Point Odyssey 数据集的结构：scene/rgb/image.jpg
            
            basenames = sorted([f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
            # 获取所有 .jpg 文件的基础名称（去掉扩展名）
            # sorted() 确保文件按顺序排列，这对视频序列很重要
            # f[:-4] 去掉文件名的最后4个字符（.jpg）
            
            num_imgs = len(basenames)
            # 当前场景的图像数量
            
            img_ids = list(np.arange(num_imgs) + offset)
            # 为当前场景的所有图像分配全局唯一的ID
            # offset 确保不同场景的图像ID不重复
            
            # ==================== 计算可用的起始图像 ====================
            cut_off = (self.num_views if not self.allow_repeat else max(self.num_views // 3, 3))
            # 计算截止值：
            # 如果不允许重复视图，截止值等于所需视图数
            # 如果允许重复视图，截止值是视图数的1/3（最少3个）
            # 这确保有足够的图像来构成一个完整的多视角序列
            
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            # 计算可以作为起始点的图像ID
            # 从第一个图像到倒数第cut_off个图像都可以作为起始点
            # 这确保从起始点开始能够采样到足够的后续图像
            
            # 注释的代码是另一种计算方式：
            # start_img_ids_ = img_ids[:-self.num_views+1]

            # ==================== 场景有效性检查 ====================
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            # 如果当前场景的图像数量少于截止值，跳过这个场景
            # 这确保每个场景都有足够的图像来构成多视角序列

            # ==================== 更新全局数据结构 ====================
            start_img_ids.extend(start_img_ids_)
            # 将当前场景的起始图像ID添加到全局列表
            
            sceneids.extend([j] * num_imgs)
            # 为当前场景的所有图像分配场景ID
            # [j] * num_imgs 创建一个长度为 num_imgs 的列表，所有元素都是 j
            
            images.extend(basenames)
            # 将当前场景的所有图像基础名称添加到全局列表
            
            scenes.append(scene)
            # 将场景名称添加到场景列表
            
            scene_img_list.append(img_ids)
            # 将当前场景的图像ID列表添加到场景图像列表

            # ==================== 更新计数器和偏移量 ====================
            offset += num_imgs
            # 更新偏移量，为下一个场景的图像ID做准备
            
            j += 1
            # 场景计数器加1

        # ==================== 保存加载的数据 ====================
        self.scenes = scenes
        # 保存实际使用的场景名称列表
        
        self.sceneids = sceneids
        # 保存每个图像对应的场景ID
        
        self.images = images
        # 保存所有图像的基础名称
        
        self.start_img_ids = start_img_ids
        # 保存所有可能的起始图像ID
        
        self.scene_img_list = scene_img_list
        # 保存每个场景的图像ID列表

    def __len__(self):
        """
        返回数据集的长度
        
        数据集的长度等于可能的起始点数量
        每个起始点可以构成一个多视角序列样本
        """
        return len(self.start_img_ids)

    def get_image_num(self):
        """
        返回数据集中图像的总数量
        
        这包括所有场景中的所有图像
        """
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        获取指定索引的多视角数据
        
        这是数据集的核心方法，负责加载和处理多视角图像数据
        
        参数:
            idx: 数据样本的索引
            resolution: 目标图像分辨率
            rng: 随机数生成器
            num_views: 需要的视图数量
            
        返回:
            views: 包含多个视图数据的列表
        """
        
        start_id = self.start_img_ids[idx]
        # 获取当前样本的起始图像ID
        
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        # 获取起始图像所属场景的所有图像ID
        # self.sceneids[start_id] 获取起始图像的场景ID
        # self.scene_img_list[场景ID] 获取该场景的所有图像ID
        
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,           # 需要的视图数量
            start_id,           # 起始图像ID
            all_image_ids,      # 场景中所有可用的图像ID
            rng,                # 随机数生成器
            max_interval=self.max_interval,  # 最大时间间隔
            video_prob=1.0,     # 视频序列概率（1.0表示总是选择时序连续的图像）
            fix_interval_prob=1.0,  # 固定间隔概率
        )
        # 这个方法（来自父类）选择一个图像序列
        # pos: 选中图像在 all_image_ids 中的位置索引
        # ordered_video: 是否是有序的视频序列
        
        image_idxs = np.array(all_image_ids)[pos]
        # 根据位置索引获取实际的图像ID
        # 将 all_image_ids 转换为 numpy 数组，然后用 pos 索引

        views = []
        # 初始化视图列表
        
        # ==================== 加载每个视图的数据 ====================
        for v, view_idx in enumerate(image_idxs):
            # 遍历选中的每个图像ID
            # v: 视图索引（0, 1, 2, ...）
            # view_idx: 图像的全局ID
            
            scene_id = self.sceneids[view_idx]
            # 获取当前图像所属的场景ID
            
            # ==================== 构建文件路径 ====================
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            # 构建场景目录路径
            # 例如：/path/to/pointodyssey/train/scene_name
            
            rgb_dir = osp.join(scene_dir, "rgb")
            # RGB图像目录
            
            depth_dir = osp.join(scene_dir, "depth")
            # 深度图目录
            
            cam_dir = osp.join(scene_dir, "cam")
            # 相机参数目录

            basename = self.images[view_idx]
            # 获取当前图像的基础名称

            # ==================== 加载RGB图像 ====================
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # 使用自定义的 imread_cv2 函数加载RGB图像
            # 文件路径：rgb_dir/basename.jpg
            
            # ==================== 加载深度图 ====================
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            # 加载对应的深度图，深度数据以 .npy 格式存储
            
            depthmap[~np.isfinite(depthmap)] = 0  # 将无效深度值设为0
            # ~np.isfinite() 找到所有非有限值（NaN、无穷大等）
            
            depthmap[depthmap > 1000] = 0.0
            # 将过大的深度值（>1000）设为0，这可能是噪声或无效测量

            # ==================== 加载相机参数 ====================
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            # 加载相机参数文件，.npz 是 numpy 的压缩格式
            
            camera_pose = cam["pose"]
            # 相机位姿（外参）：4x4变换矩阵，包含旋转和平移
            
            intrinsics = cam["intrinsics"]
            # 相机内参：3x3矩阵，包含焦距、主点等参数

            # ==================== 图像预处理 ====================
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )
            # 根据需要裁剪和缩放图像
            # 这个方法会同时调整RGB图像、深度图和相机内参
            # 确保所有数据保持一致性

            # ==================== 生成掩码 ====================
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.9, 0.05, 0.05]
            )
            # 生成图像掩码和光线掩码
            # p=[0.9, 0.05, 0.05] 可能表示不同掩码类型的概率分布
            # img_mask: 图像级别的掩码
            # ray_mask: 光线级别的掩码（用于射线投射等算法）

            # ==================== 构建视图数据字典 ====================
            views.append(
                dict(
                    img=rgb_image,                                    # RGB图像数据
                    depthmap=depthmap.astype(np.float32),            # 深度图（转换为float32）
                    camera_pose=camera_pose.astype(np.float32),      # 相机位姿（外参）
                    camera_intrinsics=intrinsics.astype(np.float32), # 相机内参
                    dataset="PointOdyssey",                          # 数据集名称标识
                    label=self.scenes[scene_id] + "_" + basename,    # 数据标签（场景名_图像名）
                    instance=osp.join(rgb_dir, basename + ".jpg"),   # 图像文件的完整路径
                    is_metric=self.is_metric,                        # 是否是度量深度
                    is_video=ordered_video,                          # 是否是视频序列
                    quantile=np.array(1.0, dtype=np.float32),       # 分位数信息（可能用于深度缩放）
                    img_mask=img_mask,                               # 图像掩码
                    ray_mask=ray_mask,                               # 光线掩码
                    camera_only=False,                               # 是否只使用相机数据
                    depth_only=False,                                # 是否只使用深度数据
                    single_view=False,                               # 是否是单视图
                    reset=False,                                     # 是否重置（可能用于序列处理）
                )
            )
            # 这个字典包含了训练深度估计和多视图几何所需的所有信息

        # ==================== 验证和返回 ====================
        assert len(views) == num_views
        # 确保返回的视图数量与请求的数量一致
        
        return views
        # 返回包含所有视图数据的列表


# ==================== 总结 ====================
"""
Point Odyssey 数据集读取器的主要功能：

1. **数据集组织**：
   - 支持多个场景，每个场景包含RGB图像、深度图和相机参数
   - 数据按 train/test/val 分割组织
   - 每个场景有不同的视角（第三人称、自我中心）

2. **多视角序列采样**：
   - 从视频序列中选择时序相关的多个视图
   - 控制视图之间的最大时间间隔
   - 确保足够的视觉重叠和连续性

3. **数据加载和预处理**：
   - 加载RGB图像、深度图和相机参数
   - 处理无效深度值和异常值
   - 根据需要调整图像分辨率
   - 生成训练所需的掩码

4. **数据格式**：
   - RGB图像：JPG格式
   - 深度图：NPY格式（numpy数组）
   - 相机参数：NPZ格式（包含pose和intrinsics）

5. **应用场景**：
   - 多视图立体视觉
   - 深度估计
   - SLAM（同时定位与建图）
   - 点跟踪和运动分析

这个数据集读取器为深度学习模型提供了标准化的接口，
使得模型可以方便地访问Point Odyssey数据集的丰富多视角信息。
""" 