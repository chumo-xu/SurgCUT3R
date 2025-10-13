#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScanNet数据集加载器 - 详细注释版本
===============================
本文件是对 scannet.py 的详细注释版本
每一行代码都有对应的中文解释，帮助理解数据集加载和处理的完整流程

ScanNet数据集加载器主要功能：
1. 继承自BaseMultiViewDataset，提供多视图数据加载能力
2. 从预处理后的ScanNet数据中加载RGB图像、深度图和相机参数
3. 支持序列化视图选择，用于视频序列训练
4. 提供数据增强和掩码生成功能

数据加载流程：
初始化 -> 加载场景元数据 -> 选择视图序列 -> 加载图像和相机数据 -> 应用变换

输入数据结构（预处理后）：
ROOT/
├── scans_train/          # 训练集
│   └── scene0000_00/     # 场景目录
│       ├── color/        # RGB图像(.jpg)
│       ├── depth/        # 深度图(.png)
│       ├── cam/          # 相机参数(.npz)
│       └── new_scene_metadata.npz  # 场景元数据
└── scans_test/           # 测试集
    └── ...
"""

# ============================================================================
# 导入必要的库 (Import Required Libraries)
# ============================================================================

import os.path as osp    # 导入路径操作模块，用于处理文件和目录路径
import cv2               # 导入OpenCV库，用于图像读取和处理
import numpy as np       # 导入NumPy库，用于数值计算和数组操作
import itertools         # 导入itertools，用于高效的迭代器操作（虽然本文件中未直接使用）
import os                # 导入os模块，用于操作系统相关的文件和目录操作
import sys               # 导入sys模块，用于系统相关的参数和函数

# 将父目录添加到Python路径中，以便导入项目内的其他模块
# osp.dirname(__file__): 获取当前文件所在的目录
# "..", "..": 向上两级目录，到达项目根目录
# 这样可以确保能够正确导入dust3r包中的其他模块
sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from tqdm import tqdm    # 导入进度条库，用于显示数据加载的进度
# 导入基础多视图数据集类，ScanNet_Multi将继承这个类
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
# 导入图像读取工具函数
from dust3r.utils.image import imread_cv2


# ============================================================================
# ScanNet多视图数据集类 (ScanNet Multi-View Dataset Class)
# ============================================================================

class ScanNet_Multi(BaseMultiViewDataset):
    """
    ScanNet多视图数据集类
    
    这个类专门用于加载和处理ScanNet数据集，支持：
    - 多视图数据加载
    - 序列化视图选择（适用于视频序列训练）
    - 相机参数和深度信息处理
    - 数据增强和掩码生成
    
    继承自BaseMultiViewDataset，具有通用的多视图数据集功能
    """
    
    def __init__(self, *args, ROOT, **kwargs):
        """
        初始化ScanNet数据集
        
        Args:
            *args: 传递给父类的位置参数
            ROOT (str): ScanNet数据集的根目录路径
            **kwargs: 传递给父类的关键字参数
        """
        # 保存数据集根目录路径，后续所有数据加载都基于这个路径
        self.ROOT = ROOT
        
        # 设置为视频模式，表示这个数据集包含时序相关的帧序列
        # 在视频模式下，会优先选择时间上连续的帧作为多视图输入
        self.video = True
        
        # 设置为度量数据集，表示深度值是真实的物理距离（米）
        # 这与相对深度不同，度量深度可以用于真实的3D重建
        self.is_metric = True
        
        # 设置最大帧间隔，限制选择的视图之间的最大时间距离
        # 值为30表示选择的帧之间最多相差30帧
        # 这有助于确保视图之间有足够的重叠和相关性
        self.max_interval = 30
        
        # 调用父类的初始化方法，传递所有参数
        # 父类会处理通用的数据集配置，如分辨率、数据增强等
        super().__init__(*args, **kwargs)

        # 加载指定分割（train/test）的数据
        # self.split是在父类中设置的，表示当前使用的数据集分割
        # _load_data方法会扫描数据目录并建立索引
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        """
        加载指定分割的数据集信息
        
        这个方法负责：
        1. 扫描数据目录，找到所有场景
        2. 读取每个场景的元数据
        3. 建立图像索引和场景映射关系
        4. 计算有效的起始帧位置
        
        Args:
            split (str): 数据集分割名称，'train' 或 'test'
        """
        # ====================================================================
        # 构建场景根目录路径 (Build Scene Root Directory Path)
        # ====================================================================
        
        # 根据分割类型选择对应的子目录
        # 训练集使用"scans_train"，测试集使用"scans_test"
        self.scene_root = osp.join(
            self.ROOT, "scans_train" if split == "train" else "scans_test"
        )
        
        # 扫描场景根目录，获取所有以"scene"开头的场景文件夹
        # listdir(): 列出目录中的所有文件和文件夹
        # startswith("scene"): 只保留场景目录，过滤其他文件
        self.scenes = [
            scene for scene in os.listdir(self.scene_root) if scene.startswith("scene")
        ]

        # ====================================================================
        # 初始化数据结构 (Initialize Data Structures)
        # ====================================================================
        
        # 图像索引偏移量，用于为每个场景的图像分配全局唯一ID
        offset = 0
        
        # 存储有效场景列表（跳过图像数量不足的场景后的结果）
        scenes = []
        
        # 存储每个图像对应的场景ID
        sceneids = []
        
        # 存储每个场景的图像ID列表
        scene_img_list = []
        
        # 存储所有图像的文件名（不包含扩展名）
        images = []
        
        # 存储每个有效起始位置的图像ID
        # 起始位置是指可以作为多视图序列开始的帧位置
        start_img_ids = []

        # 场景计数器，用于为每个有效场景分配递增的ID
        j = 0
        
        # ====================================================================
        # 遍历所有场景 (Iterate Through All Scenes)
        # ====================================================================
        
        # 使用tqdm显示处理进度，遍历所有场景
        for scene in tqdm(self.scenes):
            # 构建当前场景的完整目录路径
            scene_dir = osp.join(self.scene_root, scene)
            
            # ================================================================
            # 读取场景元数据 (Load Scene Metadata)
            # ================================================================
            
            # 从场景元数据文件中读取图像列表
            # allow_pickle=True: 允许加载包含Python对象的数组
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                # 获取当前场景中所有图像的基础文件名列表
                basenames = data["images"]
                
                # 计算当前场景的图像总数
                num_imgs = len(basenames)
                
                # 为当前场景的所有图像分配全局ID
                # np.arange(num_imgs): 创建从0到num_imgs-1的数组
                # + offset: 加上偏移量，确保全局唯一性
                img_ids = list(np.arange(num_imgs) + offset)
                
                # ============================================================
                # 计算截止点 (Calculate Cut-off Point)
                # ============================================================
                
                # 计算序列截止点，确定最少需要多少帧才能构成有效序列
                # self.allow_repeat: 是否允许重复使用帧
                cut_off = (
                    self.num_views                    # 如果不允许重复，需要num_views帧
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)  # 如果允许重复，至少需要num_views/3帧，最少3帧
                )
                
                # 计算所有可能的起始帧位置
                # 起始位置i需要满足：从i开始能够选择足够的帧（至少cut_off帧）
                # 因此最大起始位置是 num_imgs - cut_off
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                # ============================================================
                # 验证场景有效性 (Validate Scene)
                # ============================================================
                
                # 如果场景的图像数量少于截止点，跳过这个场景
                if num_imgs < cut_off:
                    print(f"Skipping {scene}")  # 打印跳过信息
                    continue  # 跳过当前场景，处理下一个

                # ============================================================
                # 添加有效场景数据 (Add Valid Scene Data)
                # ============================================================
                
                # 将当前场景的所有有效起始位置添加到全局列表
                start_img_ids.extend(start_img_ids_)
                
                # 为当前场景的每个图像分配场景ID
                # [j] * num_imgs: 创建包含num_imgs个场景ID j的列表
                sceneids.extend([j] * num_imgs)
                
                # 将当前场景的所有图像文件名添加到全局列表
                images.extend(basenames)
                
                # 将当前场景名称添加到有效场景列表
                scenes.append(scene)
                
                # 将当前场景的图像ID列表添加到场景图像列表
                scene_img_list.append(img_ids)

                # ============================================================
                # 更新计数器 (Update Counters)
                # ============================================================
                
                # 更新偏移量，为下一个场景的图像ID分配做准备
                offset += num_imgs
                
                # 递增场景计数器
                j += 1

        # ====================================================================
        # 保存处理结果 (Save Processing Results)
        # ====================================================================
        
        # 将处理后的数据保存为实例变量，供其他方法使用
        self.scenes = scenes                    # 有效场景列表
        self.sceneids = sceneids               # 每个图像对应的场景ID
        self.images = images                   # 所有图像的文件名列表
        self.start_img_ids = start_img_ids     # 所有有效起始位置
        self.scene_img_list = scene_img_list   # 每个场景的图像ID列表

    def __len__(self):
        """
        返回数据集的长度
        
        数据集的长度定义为所有可能的起始位置数量
        每个起始位置都可以生成一个多视图样本
        
        Returns:
            int: 数据集中的样本数量
        """
        return len(self.start_img_ids)

    def get_image_num(self):
        """
        获取数据集中的图像总数
        
        Returns:
            int: 数据集中包含的图像总数
        """
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        """
        获取指定索引处的多视图数据
        
        这是数据集类的核心方法，负责：
        1. 根据索引确定起始帧
        2. 选择合适的多视图序列
        3. 加载RGB图像、深度图和相机参数
        4. 应用必要的预处理和数据增强
        5. 生成掩码信息
        
        Args:
            idx (int): 样本索引
            resolution (tuple): 目标图像分辨率 (height, width)
            rng (np.random.RandomState): 随机数生成器
            num_views (int): 需要的视图数量
            
        Returns:
            list: 包含num_views个字典的列表，每个字典包含一个视图的所有信息
        """
        # ====================================================================
        # 确定起始帧和图像序列 (Determine Start Frame and Image Sequence)
        # ====================================================================
        
        # 根据样本索引获取对应的起始图像ID
        start_id = self.start_img_ids[idx]
        
        # 获取起始图像所属场景的所有图像ID列表
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        
        # 从起始位置选择合适的多视图序列
        # get_seq_from_start_id是父类方法，实现智能的视图选择策略
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,              # 需要选择的视图数量
            start_id,               # 起始图像ID
            all_image_ids,          # 当前场景的所有图像ID
            rng,                    # 随机数生成器
            max_interval=self.max_interval,     # 最大帧间隔
            video_prob=0.6,         # 选择连续视频序列的概率
            fix_interval_prob=0.6,  # 使用固定间隔的概率
            block_shuffle=16,       # 块内随机打乱的大小
        )
        
        # 将选择的位置转换为实际的图像索引
        image_idxs = np.array(all_image_ids)[pos]

        # ====================================================================
        # 加载每个视图的数据 (Load Data for Each View)
        # ====================================================================
        
        # 初始化视图列表，用于存储所有视图的数据
        views = []
        
        # 遍历选择的每个视图
        for v, view_idx in enumerate(image_idxs):
            # ================================================================
            # 确定文件路径 (Determine File Paths)
            # ================================================================
            
            # 获取当前视图所属的场景ID
            scene_id = self.sceneids[view_idx]
            
            # 构建当前场景的目录路径
            scene_dir = osp.join(self.scene_root, self.scenes[scene_id])
            
            # 构建各类数据的子目录路径
            rgb_dir = osp.join(scene_dir, "color")    # RGB图像目录
            depth_dir = osp.join(scene_dir, "depth")  # 深度图目录
            cam_dir = osp.join(scene_dir, "cam")      # 相机参数目录

            # 获取当前图像的基础文件名（不包含扩展名）
            basename = self.images[view_idx]

            # ================================================================
            # 加载RGB图像 (Load RGB Image)
            # ================================================================
            
            # 使用自定义的图像读取函数加载RGB图像
            # imread_cv2是项目中的工具函数，可能包含特殊的预处理
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            
            # ================================================================
            # 加载深度图 (Load Depth Map)
            # ================================================================
            
            # 加载深度图，使用IMREAD_UNCHANGED保持原始数据类型
            depthmap = imread_cv2(
                osp.join(depth_dir, basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            
            # 将深度值从毫米转换为米
            # ScanNet的深度图通常以毫米为单位存储，除以1000转换为米
            depthmap = depthmap.astype(np.float32) / 1000
            
            # 处理无效深度值
            # ~np.isfinite(): 找到所有非有限值（NaN, inf等）
            # 将这些无效值设置为0
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            # ================================================================
            # 加载相机参数 (Load Camera Parameters)
            # ================================================================
            
            # 从.npz文件中加载相机参数
            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            
            # 提取相机位姿矩阵（4x4变换矩阵）
            camera_pose = cam["pose"]
            
            # 提取相机内参矩阵（3x3）
            intrinsics = cam["intrinsics"]
            
            # ================================================================
            # 应用裁剪和缩放 (Apply Cropping and Resizing)
            # ================================================================
            
            # 根据目标分辨率对图像、深度图和内参进行必要的裁剪和缩放
            # _crop_resize_if_necessary是父类方法，处理分辨率适配
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,    # 输入RGB图像
                depthmap,     # 输入深度图
                intrinsics,   # 输入相机内参
                resolution,   # 目标分辨率
                rng=rng,      # 随机数生成器（用于随机裁剪）
                info=view_idx # 额外信息（用于调试）
            )

            # ================================================================
            # 生成掩码 (Generate Masks)
            # ================================================================
            
            # 生成图像掩码和射线掩码，用于训练时的采样和损失计算
            # get_img_and_ray_masks是父类方法，实现各种掩码策略
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric,           # 是否为度量数据集
                v,                        # 当前视图索引
                rng,                      # 随机数生成器
                p=[0.75, 0.2, 0.05]      # 不同掩码类型的概率分布
            )

            # ================================================================
            # 构建视图字典 (Build View Dictionary)
            # ================================================================
            
            # 为当前视图创建包含所有必要信息的字典
            views.append(
                dict(
                    # 基础数据
                    img=rgb_image,                              # RGB图像数据
                    depthmap=depthmap.astype(np.float32),      # 深度图数据
                    camera_pose=camera_pose.astype(np.float32), # 相机位姿矩阵
                    camera_intrinsics=intrinsics.astype(np.float32), # 相机内参矩阵
                    
                    # 元数据
                    dataset="ScanNet",                          # 数据集名称
                    label=self.scenes[scene_id] + "_" + basename, # 视图标签（场景名_图像名）
                    instance=f"{str(idx)}_{str(view_idx)}",     # 实例标识符
                    
                    # 数据集属性
                    is_metric=self.is_metric,                   # 是否为度量深度
                    is_video=ordered_video,                     # 是否为有序视频序列
                    quantile=np.array(0.98, dtype=np.float32), # 深度值的分位数（用于归一化）
                    
                    # 掩码信息
                    img_mask=img_mask,                          # 图像掩码
                    ray_mask=ray_mask,                          # 射线掩码
                    
                    # 训练标志
                    camera_only=False,                          # 是否只使用相机参数
                    depth_only=False,                           # 是否只使用深度信息
                    single_view=False,                          # 是否为单视图模式
                    reset=False,                                # 是否重置状态
                )
            )
        
        # ====================================================================
        # 验证和返回结果 (Validate and Return Results)
        # ====================================================================
        
        # 确保返回的视图数量与请求的数量一致
        assert len(views) == num_views
        
        # 返回包含所有视图信息的列表
        return views


# ============================================================================
# 使用说明和数据流程 (Usage Instructions and Data Flow)
# ============================================================================

"""
ScanNet数据集加载器使用说明：

1. 初始化：
   dataset = ScanNet_Multi(
       ROOT="/path/to/processed/scannet",
       split="train",
       num_views=3,
       resolution=(224, 224)
   )

2. 数据加载：
   views = dataset[0]  # 获取第一个样本的多视图数据
   
3. 数据结构：
   每个sample包含num_views个视图，每个视图包含：
   - img: RGB图像 (H, W, 3)
   - depthmap: 深度图 (H, W)
   - camera_pose: 相机位姿 (4, 4)
   - camera_intrinsics: 相机内参 (3, 3)
   - 各种元数据和掩码信息

4. 视图选择策略：
   - 优先选择时间连续的帧（video_prob=0.6）
   - 限制帧间最大距离（max_interval=30）
   - 支持固定间隔和随机采样
   - 处理重复和非重复模式

5. 数据预处理：
   - 深度值单位转换（毫米→米）
   - 图像缩放和裁剪
   - 相机参数调整
   - 掩码生成

关键特性：
- 支持大规模ScanNet数据集的高效加载
- 智能的多视图序列选择
- 完整的相机几何信息
- 灵活的数据增强和掩码策略
- 与训练流程的无缝集成
""" 